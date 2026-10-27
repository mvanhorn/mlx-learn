"""Fresh-fit wide logistic comparison against a frozen native implementation.

    python -m benchmarks.logistic_wide --dtype both --profile full \
        --output bench/logistic-wide.json

The default source-only reference archive and SHA256 manifest are checked into
``benchmarks/reference``. Extraction validates every path and content hash in a
temporary directory. An explicit ``--reference-dir`` can instead select a local
snapshot containing ``mlxlearn`` and ``manifest.json``. The complete reference is
loaded under an isolated package name, including validation and preprocessing.
Nothing is patched in sklearn. Run this standalone, not inside a threaded host.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import hashlib
import importlib
import json
import math
import stat
import sys
import tempfile
import time
import types
import warnings
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import mlx.core as mx
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from mlxlearn import config_context
from mlxlearn._common.config import tuning_context
from mlxlearn._solvers import logistic, native_logistic
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.linear_model import _logistic as estimator_module

from .logistic_gpu import Case, make_data, quality, record_environment, summarize
from .logistic_native import state_metadata

VARIANTS = ("previous_native", "optimized_gpu", "sklearn")
DEFAULT_REFERENCE_ARCHIVE = (
    Path(__file__).resolve().parent / "reference" / "native-logistic-2026-10-26.zip"
)
REFERENCE_KERNEL_PREFIX = "wide_reference_"
PROFILES = {
    "smoke": (
        Case("wide_20k_1024", 20_000, 1_024),
        Case("wide_60k_1024", 60_000, 1_024),
        Case("tall_control", 500_000, 32),
    ),
    "full": (
        Case("wide_20k_512", 20_000, 512),
        Case("wide_20k_1024", 20_000, 1_024),
        Case("wide_60k_1024", 60_000, 1_024),
        Case("wide_40k_2048", 40_000, 2_048),
        Case("wide_20k_4096", 20_000, 4_096),
        Case("wide_weighted", 20_000, 1_024, weighted=True),
        Case("tall_control", 500_000, 32),
    ),
}


def package_source_paths():
    """Include native source as well as Python when recording solver provenance."""
    package_root = Path(logistic.__file__).resolve().parents[1]
    extensions = {".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".m", ".mm", ".metal"}
    return [
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (path.suffix in extensions or path.name == "py.typed")
    ]


def loaded_native_extension():
    """Describe the extension actually loaded by the live native-loop wrapper."""
    command = getattr(native_logistic, "native_command", None)
    extension = getattr(command, "_metal_loop", None)
    if extension is None:
        return None
    name = extension.__name__
    if sys.modules.get(name) is not extension:
        raise RuntimeError(f"Native extension is not the registered loaded module: {name}")
    path = Path(extension.__file__).resolve()
    return {"module": name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@contextlib.contextmanager
def reference_sources(directory=None, archive=None):
    """Validate and materialize only the source files enumerated by the manifest."""
    if directory is not None:
        directory = Path(directory).resolve()
        yield directory, {"directory": str(directory)}
        return
    archive = Path(archive or DEFAULT_REFERENCE_ARCHIVE).resolve()
    manifest_path = archive.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_hash != manifest["archive"]["sha256"]:
        raise RuntimeError(f"Reference archive hash mismatch: {archive}")
    expected_files = manifest["files"]
    with tempfile.TemporaryDirectory(prefix="mlxlearn-wide-reference-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or set(names) != set(expected_files):
                raise RuntimeError("Reference archive entries differ from the manifest")
            for entry in entries:
                relative = PurePosixPath(entry.filename)
                mode = entry.external_attr >> 16
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in entry.filename
                    or relative.as_posix() != entry.filename
                    or not relative.parts
                    or relative.parts[0] != "mlxlearn"
                    or (relative.suffix != ".py" and relative.name != "py.typed")
                    or entry.is_dir()
                    or not stat.S_ISREG(mode)
                ):
                    raise RuntimeError(f"Unsafe reference archive entry: {entry.filename}")
                data = source.read(entry)
                if hashlib.sha256(data).hexdigest() != expected_files[entry.filename]:
                    raise RuntimeError(f"Reference source hash mismatch: {entry.filename}")
                destination = root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        (root / "manifest.json").write_text(json.dumps(manifest))
        yield (
            root,
            {
                "archive": str(archive),
                "archive_sha256": archive_hash,
                "manifest_path": str(manifest_path),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
        )


@contextlib.contextmanager
def frozen_reference(directory=None, *, archive=None, extension=None):
    """Load a complete immutable source snapshot without replacing live modules."""
    with reference_sources(directory, archive) as (source_directory, provenance):
        with _loaded_reference(source_directory, provenance, extension=extension) as reference:
            yield reference


@contextlib.contextmanager
def reference_kernel_namespace(prefix=REFERENCE_KERNEL_PREFIX):
    """Give frozen Metal functions distinct symbols without editing their source.

    This temporary factory wrapper is used only during eager reference imports
    and restored before fitting. The standalone benchmark has no other threads
    importing kernels. Isolation does not rely on MLX's kernel-cache key details.
    """
    original = mx.fast.metal_kernel
    names = []

    def named_kernel(name, *args, **kwargs):
        name = prefix + name
        names.append(name)
        return original(name, *args, **kwargs)

    mx.fast.metal_kernel = named_kernel
    try:
        yield names
    finally:
        mx.fast.metal_kernel = original


@contextlib.contextmanager
def _loaded_reference(directory, provenance, *, extension=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Reference snapshot hash mismatch: {path}")
    name = "_mlxlearn_wide_reference"
    if name in sys.modules:
        raise RuntimeError("A reference package is already loaded")
    package = types.ModuleType(name)
    package.__path__ = [str(directory / "mlxlearn")]
    package.__package__ = name
    sys.modules[name] = package
    try:
        # A source snapshot may require its own native command-loop extension.
        # Load a separately copied binary under the reference package name so
        # it cannot silently use the live extension or fall back to MLX.
        expected_binary = manifest.get("native_binary_sha256")
        if expected_binary is not None:
            if extension is None:
                raise RuntimeError("This reference requires --reference-extension")
            extension_path = Path(extension).resolve()
            if hashlib.sha256(extension_path.read_bytes()).hexdigest() != expected_binary:
                raise RuntimeError("Reference native extension hash mismatch")
            importlib.import_module(f"{name}._solvers")
            qualified = f"{name}._solvers._metal_loop"
            spec = importlib.util.spec_from_file_location(qualified, extension_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot load the reference native extension")
            binary = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = binary
            spec.loader.exec_module(binary)
            provenance = {**provenance, "native_extension": str(extension_path),
                          "native_extension_sha256": expected_binary}
        elif extension is not None:
            raise RuntimeError("Reference manifest does not identify a native extension")
        # Distinct snapshots need distinct symbols even if loaded successively
        # in one process, since their kernel source may differ.
        source_identity = hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True).encode()
        ).hexdigest()[:12]
        kernel_prefix = REFERENCE_KERNEL_PREFIX + source_identity + "_"
        kernel_modules = []
        for relative in manifest["files"]:
            path = PurePosixPath(relative)
            if path.suffix != ".py":
                continue
            tree = ast.parse((directory / relative).read_text())
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "metal_kernel"
                for node in ast.walk(tree)
            ):
                parts = path.with_suffix("").parts[1:]
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                kernel_modules.append(".".join(parts))
        with reference_kernel_namespace(kernel_prefix) as kernel_names:
            config = importlib.import_module(f"{name}._common.config")
            estimator = importlib.import_module(f"{name}.linear_model._logistic")
            solver = importlib.import_module(f"{name}._solvers.logistic")
            native = importlib.import_module(f"{name}._solvers.native_logistic")
            # Some helpers, including the speculative gradient and input
            # validation kernels, are normally imported only on the first fit.
            # Import them while the factory is wrapped; numerical code stays
            # byte-for-byte identical and no kernel is evaluated here.
            for module in kernel_modules:
                importlib.import_module(f"{name}.{module}")
        provenance = {
            **provenance,
            "kernel_name_prefix": kernel_prefix,
            "namespaced_kernels": kernel_names,
            "eager_kernel_modules": kernel_modules,
        }
        yield types.SimpleNamespace(
            config=config,
            estimator=estimator,
            solver=solver,
            native=native,
            manifest=manifest,
            directory=directory,
            provenance=provenance,
        )
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


@contextlib.contextmanager
def observed_implementation(estimator, solver, native):
    """Observe actual native command completion, failure, and MLX resumption.

    Historical reference packages returned an LBFGSState or None; modern
    packages can instead return a NativeAttempt containing the native state.
    Count only a converged native result as completed. A failed NativeAttempt
    whose final solver state is a new object indicates that the MLX fallback
    was entered; compare iteration counts to distinguish accepted progress
    from a resumption that made no accepted step.
    """
    original_solve = estimator.solve_logistic
    original_native = native.minimize_binary
    original_fused = solver.minimize
    from mlxlearn_device import _newton

    original_newton = _newton.minimize_newton
    command_module = getattr(native, "native_command", None)
    original_command = getattr(command_module, "run_native_loop", None)
    attempt_type = getattr(command_module, "NativeAttempt", None)
    records = {"solutions": [], "solver_calls": [], "native_loop_calls": [],
               "native_loop_attempts": [], "newton_calls": []}
    # Actual state objects are held only locally, never serialized into results.
    pending_attempts = []

    def solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        records["solutions"].append(state_metadata(result))
        return result

    def native_fit(*args, **kwargs):
        before = len(records["native_loop_calls"])
        pending_before = len(pending_attempts)
        result = original_native(*args, **kwargs)
        for index, attempt in pending_attempts[pending_before:]:
            state = attempt.state
            if state.converged:
                outcome = "completed"
            elif result is state:
                outcome = "terminal_nonconvergence"
            elif result.n_iter > state.n_iter:
                outcome = "resumed"
            else:
                outcome = "resumed_no_progress"
            records["native_loop_calls"][index] = outcome
            records["native_loop_attempts"].append({
                "status": attempt.status,
                "native_n_iter": state.n_iter,
                "final_n_iter": result.n_iter,
                "native_converged": state.converged,
                "outcome": outcome,
            })
        command_calls = records["native_loop_calls"][before:]
        if any(call in {"resumed", "resumed_no_progress"} for call in command_calls):
            implementation = "native_then_mlx"
        elif any(call in {"completed", "terminal_nonconvergence"} for call in command_calls):
            implementation = "native_command_loop"
        else:
            implementation = "native_metal"
        records["solver_calls"].append({"implementation": implementation, **state_metadata(result)})
        return result

    def command_loop(*args, **kwargs):
        result = original_command(*args, **kwargs)
        if attempt_type is not None and isinstance(result, attempt_type):
            pending_attempts.append((len(records["native_loop_calls"]), result))
            records["native_loop_calls"].append("attempted")
        else:
            # None is ambiguous for older reference packages: it may describe
            # ineligibility or a failed loop, so retain their legacy label.
            records["native_loop_calls"].append("completed" if result is not None else "fallback")
        return result

    def fused_fit(*args, **kwargs):
        result = original_fused(*args, **kwargs)
        records["solver_calls"].append({"implementation": "fused_mlx", **state_metadata(result)})
        return result

    def newton_fit(*args, **kwargs):
        result = original_newton(*args, **kwargs)
        records["newton_calls"].append({
            "n_iter": result.n_iter, "n_fev": result.n_fev,
            "converged": result.converged, "stop_reason": result.stop_reason,
            "objective": result.objective, "grad_max_abs": result.gradient_max_abs,
            "relative_duality_gap": result.relative_gap, "hvp_count": result.hvp_count,
            "budget": kwargs["max_iter"], "resumed": kwargs.get("initial") is not None,
        })
        return result

    estimator.solve_logistic = solve
    native.minimize_binary = native_fit
    solver.minimize = fused_fit
    _newton.minimize_newton = newton_fit
    if original_command is not None:
        command_module.run_native_loop = command_loop
    try:
        yield records
    finally:
        estimator.solve_logistic = original_solve
        native.minimize_binary = original_native
        solver.minimize = original_fused
        _newton.minimize_newton = original_newton
        if original_command is not None:
            command_module.run_native_loop = original_command


def fit_once(variant, X, y, weights, *, reference, max_iter, tol):
    if variant == "previous_native":
        cls = reference.estimator.LogisticRegression
        settings = reference.config.config_context
        tuning = reference.config.tuning_context
        observed = observed_implementation(reference.estimator, reference.solver, reference.native)
    else:
        cls = SkLogisticRegression if variant == "sklearn" else LogisticRegression
        settings, tuning = config_context, tuning_context
        observed = observed_implementation(estimator_module, logistic, native_logistic)
    with (
        settings(device="auto"),
        tuning(force_backend="" if variant == "sklearn" else "mlx"),
        observed as records,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        mx.synchronize()
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            start = time.perf_counter()
            model = cls(C=1.0, tol=tol, max_iter=max_iter).fit(X, y, sample_weight=weights)
            mx.synchronize()
            duration = time.perf_counter() - start
        finally:
            if gc_enabled:
                gc.enable()
    backend = getattr(model, "_execution_backend_", "sklearn")
    if variant == "sklearn":
        assert not records["solutions"] and not records["solver_calls"]
    metadata = {
        "backend": backend,
        "n_iter": model.n_iter_.tolist(),
        "implementation": records["solver_calls"][-1]["implementation"]
        if records["solver_calls"]
        else "sklearn",
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)} for item in caught
        ],
        **records,
    }
    return duration, model, metadata


def run_case(
    case, *, reference, seed, dtype, warmup, repeats, max_iter, tol,
    require_native_loop=False, require_reference_native_loop=False, audit_every_fit=False,
):
    X, y, weights = make_data(case, seed, dtype=dtype)
    first, models = {}, {}
    optimized_models = []
    samples = {key: [] for key in VARIANTS}
    details = {key: [] for key in VARIANTS}
    for index in range(warmup):
        for variant in VARIANTS:
            elapsed, _, metadata = fit_once(
                variant, X, y, weights, reference=reference, max_iter=max_iter, tol=tol
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
                variant, X, y, weights, reference=reference, max_iter=max_iter, tol=tol
            )
            samples[variant].append(elapsed)
            details[variant].append(metadata)
            models[variant] = model
            if variant == "optimized_gpu" and audit_every_fit:
                optimized_models.append(model)
    timings = {key: summarize(values) for key, values in samples.items()}
    scores = {
        key: quality(model, models["sklearn"], X, y, weights) for key, model in models.items()
    }
    for score in scores.values():
        score["float64_gradient_within_requested_tol"] = score["gradient_max_abs_float64"] <= tol
    result = {
        "case": asdict(case),
        "dtype": dtype,
        "seed": seed,
        "parameters": {
            "C": 1.0,
            "tol": tol,
            "max_iter": max_iter,
            "fit_intercept": True,
            "require_native_loop": require_native_loop,
            "require_reference_native_loop": require_reference_native_loop,
            "audit_every_fit": audit_every_fit,
        },
        "first_case_calls_not_process_cold": first,
        "timed_orders": orders,
        "timings": timings,
        "fit_details": details,
        "quality_last_fit": scores,
        "speedup_over_previous_native": timings["previous_native"]["median_ms"]
        / timings["optimized_gpu"]["median_ms"],
        "speedup_over_sklearn": timings["sklearn"]["median_ms"]
        / timings["optimized_gpu"]["median_ms"],
    }
    if audit_every_fit:
        # Keep verification after every timed fit has finished: checking a model
        # invokes prediction and expensive float64 operations, neither of which
        # may contaminate the next timed fit. Fitted models retain only their
        # coefficients, not their training matrix.
        result["quality_each_fit"] = [
            quality(model, models["sklearn"], X, y, weights)
            for model in optimized_models[:-1]
        ] + [scores["optimized_gpu"]]
    result["quality_audit"] = audit_quality(result)
    return result


def audit_quality(result):
    """Audit every optimized repeat and independent accuracy outside the timer.

    This is deliberately separate from timing and does not discard failed
    measurements. A benchmark may finish and save useful evidence while failing
    the quality gate.
    """
    tolerance = result["parameters"]["tol"]
    failures = []

    def fail(check, detail, repeat=None):
        item = {"check": check, "detail": detail}
        if repeat is not None:
            item["repeat"] = repeat
        failures.append(item)

    details = result["fit_details"]["optimized_gpu"]
    expected_fits = len(result["timings"]["optimized_gpu"]["samples_s"])
    if not details:
        fail("measured_fits", "No measured optimized GPU fits were recorded.")
    if len(details) != expected_fits:
        fail("measured_fits", f"Expected {expected_fits} fit records, found {len(details)}.")
    if result["parameters"].get("require_reference_native_loop"):
        reference_details = result["fit_details"].get("previous_native", [])
        if len(reference_details) != expected_fits:
            fail("reference_native_loop", "Missing measured reference fit records.")
        for index, fit in enumerate(reference_details):
            if fit["implementation"] != "native_command_loop":
                fail("reference_native_loop", "Reference did not complete the native loop.", index)
    for index, fit in enumerate(details):
        implementation = fit["implementation"]
        if fit["backend"] != "mlx" or implementation not in {"native_metal", "native_command_loop", "native_then_mlx"}:
            fail("gpu_implementation", "Fit did not use the native GPU implementation.", index)
        if implementation == "native_command_loop" and "completed" not in fit.get("native_loop_calls", []):
            fail("gpu_implementation", "No completed native command loop was observed.", index)
        if result["parameters"].get("require_native_loop") and implementation != "native_command_loop":
            fail("native_loop_required", "Fit did not complete the native command loop.", index)
        if fit["warnings"]:
            fail("warnings", fit["warnings"], index)
        if len(fit["solutions"]) != 1 or not fit["solver_calls"]:
            fail("solver_metadata", "Missing solution or solver convergence metadata.", index)
        elif fit["solver_calls"][-1]["implementation"] != implementation:
            fail("gpu_implementation", "Fit and solver route metadata disagree.", index)
        for state in fit["solutions"] + fit["solver_calls"]:
            gradient = state["grad_max_abs"]
            if not state["converged"] or state["stop_reason"] != "gtol":
                fail("convergence", f"Solver stopped with {state['stop_reason']!r}.", index)
            if not math.isfinite(gradient) or gradient > tolerance:
                fail("reported_gradient", f"Gradient {gradient!r} exceeds {tolerance!r}.", index)
            if not math.isfinite(state["objective"]):
                fail("reported_objective", "Objective is not finite.", index)

    every_fit = result["parameters"].get("audit_every_fit", False)
    scores = result.get("quality_each_fit", []) if every_fit else [result["quality_last_fit"]["optimized_gpu"]]
    if every_fit and len(scores) != expected_fits:
        fail("independent_quality_fits", f"Expected {expected_fits} independent scores, found {len(scores)}.")
    for index, score in enumerate(scores):
        repeat = index if every_fit else None
        gradient = score["gradient_max_abs_float64"]
        if not math.isfinite(gradient) or gradient > tolerance:
            fail("float64_gradient", f"Independent gradient {gradient!r} exceeds {tolerance!r}.", repeat)
        gap = score["objective_relative_gap_to_sklearn"]
        if not math.isfinite(score["objective_float64"]) or not math.isfinite(gap) or gap > 1e-5:
            fail(
                "objective_accuracy",
                f"Relative objective increase {gap!r} exceeds 1e-5 or is nonfinite.",
                repeat,
            )
        agreement = score["prediction_agreement_sklearn"]
        if not math.isfinite(agreement) or not 0.995 <= agreement <= 1.0:
            fail(
                "prediction_agreement",
                f"Prediction agreement {agreement!r} is below 0.995 or invalid.", repeat,
            )
    return {
        "passed": not failures,
        "measured_fits_checked": len(details),
        "independent_quality_fits_checked": len(scores),
        "thresholds": {
            "gradient": tolerance,
            "objective_relative_increase": 1e-5,
            "prediction_agreement": 0.995,
        },
        "failures": failures,
    }


def audit_sources(payload, reference):
    """Reject a run if its current or frozen source changed while timing."""
    expected = dict(payload["environment"]["source_sha256"])
    expected.update(
        {
            str(reference.directory / path): digest
            for path, digest in reference.manifest["files"].items()
        }
    )
    provenance = reference.provenance
    if "native_extension" in provenance:
        expected[provenance["native_extension"]] = provenance["native_extension_sha256"]
    extension = payload["environment"].get("native_extension")
    if extension is not None:
        expected[extension["path"]] = extension["sha256"]
    if "archive" in provenance:
        expected[provenance["archive"]] = provenance["archive_sha256"]
        expected[provenance["manifest_path"]] = provenance["manifest_sha256"]
    failures = []
    for filename, digest in expected.items():
        path = Path(filename)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(filename)
    if extension is not None:
        try:
            if loaded_native_extension() != extension:
                failures.append("loaded native extension changed during measurement")
        except (OSError, RuntimeError):
            failures.append("loaded native extension is unavailable after measurement")
    used_loop = any(
        fit["implementation"] in {"native_command_loop", "native_then_mlx"}
        for result in payload.get("results", [])
        for fit in result["fit_details"]["optimized_gpu"]
    )
    if used_loop and extension is None:
        failures.append("native command loop completed without captured extension provenance")
    return {"passed": not failures, "files_checked": len(expected), "changed_or_missing": failures}


def update_audit_summary(payload, *, complete=False, source_audit=None):
    failed_cases = [
        f"{result['dtype']}/{result['case']['name']}"
        for result in payload["results"]
        if not result["quality_audit"]["passed"]
    ]
    source_passed = source_audit is not None and source_audit["passed"]
    payload["quality_audit"] = {
        "status": "incomplete"
        if not complete
        else "pass"
        if not failed_cases and source_passed
        else "fail",
        "passed": None if not complete else not failed_cases and source_passed,
        "completed_cases": len(payload["results"]),
        "failed_cases": failed_cases,
        "source_integrity": source_audit,
    }


def write_report(path, payload):
    provenance = payload["reference"]
    reference_name = Path(provenance.get("archive", provenance.get("directory", "reference"))).name
    lines = [
        "# Wide native logistic regression benchmark",
        "",
        f"Measured {payload['measured_at_utc']}; {payload['repeats']} fresh fits after "
        f"{payload['warmup']} warmup(s), rotated variant order, both MLX paths forced to GPU.",
        "",
        f"Frozen reference: `{reference_name}`. Its manifest and source hashes are recorded in JSON.",
        "",
        "The previous-native baseline loads the complete frozen package, including its original "
        "validation, conversion, centering, optimizer, and direct convergence verification. "
        "The new implementation uses the current package. Stock sklearn is unpatched. "
        "Generation and float64 quality checks are outside the timer; full estimator "
        "construction and fit are inside it. Every fit starts from zero without input or fitted-state caches.",
        "Frozen Metal kernel symbols receive a separate runtime prefix during module import; "
        "verified source files and numerical expressions remain unchanged. The original kernel "
        "factory is restored before fitting, and the prefix is recorded in JSON.",
        "",
        "| Input | Case | Previous native ms | Optimized ms | sklearn ms | vs previous | vs sklearn | Native loop completed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        case, timing = result["case"], result["timings"]
        fits = result["fit_details"]["optimized_gpu"]
        loop_count = sum(fit["implementation"] == "native_command_loop" for fit in fits)
        lines.append(
            f"| {result['dtype']} | {case['name']} ({case['n_samples']:,} × {case['n_features']:,}) | "
            f"{timing['previous_native']['median_ms']:.3f} | {timing['optimized_gpu']['median_ms']:.3f} | "
            f"{timing['sklearn']['median_ms']:.3f} | {result['speedup_over_previous_native']:.2f}× | "
            f"{result['speedup_over_sklearn']:.2f}× | {loop_count}/{len(fits)} |"
        )
    lines += [
        "",
        "Ratios use warmed medians. The JSON includes minimum/maximum/IQR, all raw samples, "
        "first calls, source hashes, execution order, backend/implementation, accepted iterations, "
        "objective evaluations, stop reasons, warnings, and independent float64 objectives/gradients. "
        "First calls are reported but are not isolated process-cold measurements.",
        "The native-loop count records completed loop executions; native-to-MLX resumptions "
        "are recorded separately in JSON. Legacy attempts returning fallback "
        "remain visible in each fit's JSON metadata. Use --require-native-loop for cases where "
        "the command loop must run on every measured fit. --audit-every-fit independently "
        "checks all optimized fits after timing, instead of only the last fit in each case.",
        "",
        f"**Quality/source audit: {payload['quality_audit']['status'].upper()}.** "
        "Every optimized repeat must report native GPU gradient convergence without warnings. "
        "Independent float64 gradient must meet the requested tolerance, objective increase "
        "must be at most 1e-5 relatively, and prediction agreement must be at least 99.5%. "
        "Current and frozen source hashes must remain unchanged through the run. Failed "
        "rows remain in this report and JSON; a completed failed audit exits nonzero.",
        "",
        "| Input / case | New max gradient (float64) | Objective gap vs sklearn | Prediction agreement |",
        "|---|---:|---:|---:|",
    ]
    for result in payload["results"]:
        score = result["quality_last_fit"]["optimized_gpu"]
        lines.append(
            f"| {result['dtype']} / {result['case']['name']} | {score['gradient_max_abs_float64']:.6g} | "
            f"{score['objective_relative_gap_to_sklearn']:.6g} | {score['prediction_agreement_sklearn']:.6f} |"
        )
        if not result["quality_audit"]["passed"]:
            reasons = sorted({failure["check"] for failure in result["quality_audit"]["failures"]})
            lines.append(f"\nFailed audit for this row: {', '.join(reasons)}.\n")
    lines += [
        "",
        "The original feature gradient is independent of the optimizer's centered gradient. "
        "All warnings and any failure to meet the requested tolerance remain visible in JSON. "
        "Threadpoolctl can limit recognized pools; Apple Accelerate may not honor that limit.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference-dir", type=Path)
    reference.add_argument("--reference-archive", type=Path)
    parser.add_argument(
        "--reference-extension", type=Path,
        help="hash-verified native extension for a source snapshot that requires it",
    )
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--dtype", choices=("float64", "float32", "both"), default="both")
    parser.add_argument("--case", action="append")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--require-native-loop", action="store_true",
        help="fail the audit if any optimized fit falls back from the native command loop",
    )
    parser.add_argument(
        "--require-reference-native-loop", action="store_true",
        help="fail the audit if any frozen-reference fit falls back from its native loop",
    )
    parser.add_argument(
        "--audit-every-fit", action="store_true",
        help="independently check every optimized fit after timing, instead of just the last",
    )
    parser.add_argument("--output", type=Path, default=Path("bench/logistic-wide.json"))
    parser.add_argument("--report", type=Path, default=Path("bench/logistic-wide.md"))
    args = parser.parse_args(argv)
    if args.repeats < 3 or args.warmup < 1:
        parser.error("use at least three repetitions and one warmup")
    if args.max_iter < 1 or args.tol < 0 or not math.isfinite(args.tol):
        parser.error("max-iter must be positive; tol must be finite and nonnegative")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    cases = PROFILES[args.profile]
    if args.case:
        unknown = set(args.case) - {case.name for case in cases}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases = tuple(case for case in cases if case.name in args.case)
    dtypes = ("float64", "float32") if args.dtype == "both" else (args.dtype,)
    with (
        frozen_reference(
            args.reference_dir, archive=args.reference_archive, extension=args.reference_extension
        ) as reference,
        threadpool_limits(limits=args.threads),
    ):
        payload = {
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile": args.profile,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "requested_thread_limit": args.threads,
            "input_dtypes": dtypes,
            "reference": {
                **reference.provenance,
                "manifest": reference.manifest,
                "boundary": "complete frozen package including preprocessing",
            },
            "environment": record_environment(),
            "variants": VARIANTS,
            "results": [],
        }
        root = Path(__file__).resolve().parents[1]
        build_files = [root / filename for filename in ("setup.py", "pyproject.toml")]
        paths = (
            [Path(__file__), Path(state_metadata.__code__.co_filename)]
            + package_source_paths()
            + [path for path in build_files if path.is_file()]
        )
        payload["environment"]["native_extension"] = loaded_native_extension()
        payload["environment"]["source_sha256"].update(
            {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        )
        for dtype in dtypes:
            for case in cases:
                print(
                    f"Running {dtype}/{case.name}: {case.n_samples:,} × {case.n_features:,}",
                    flush=True,
                )
                result = run_case(
                    case,
                    reference=reference,
                    seed=args.seed,
                    dtype=dtype,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    require_native_loop=args.require_native_loop,
                    require_reference_native_loop=args.require_reference_native_loop,
                    audit_every_fit=args.audit_every_fit,
                )
                payload["results"].append(result)
                update_audit_summary(payload)
                print(
                    {key: round(value["median_ms"], 3) for key, value in result["timings"].items()},
                    flush=True,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                write_report(args.report, payload)
        update_audit_summary(payload, complete=True, source_audit=audit_sources(payload, reference))
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        write_report(args.report, payload)
        print(f"Quality/source audit: {payload['quality_audit']['status'].upper()}", flush=True)
    return 0 if payload["quality_audit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
