"""CPU-only checks for benchmark evidence gates; no estimator fits or GPU work."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks import logistic_gpu, logistic_profile, logistic_wide


def valid_result():
    state = {"converged": True, "stop_reason": "gtol", "grad_max_abs": 9e-5, "objective": 0.2}
    fit = {
        "backend": "mlx",
        "implementation": "native_metal",
        "warnings": [],
        "solutions": [state],
        "solver_calls": [{**state, "implementation": "native_metal"}],
    }
    return {
        "dtype": "float32",
        "case": {"name": "test"},
        "parameters": {"tol": 1e-4},
        "timings": {"optimized_gpu": {"samples_s": [0.1, 0.1]}},
        "fit_details": {"optimized_gpu": [copy.deepcopy(fit), copy.deepcopy(fit)]},
        "quality_last_fit": {
            "optimized_gpu": {
                "gradient_max_abs_float64": 9e-5,
                "objective_float64": 0.2,
                "objective_relative_gap_to_sklearn": -1e-6,
                "prediction_agreement_sklearn": 0.999,
            }
        },
    }


def test_quality_audit_accepts_convergence_and_improved_objective():
    audit = logistic_wide.audit_quality(valid_result())
    assert audit["passed"]
    assert audit["measured_fits_checked"] == 2


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("backend", "cpu", "gpu_implementation"),
        ("implementation", "fused_mlx", "gpu_implementation"),
        ("warnings", [{"message": "ConvergenceWarning"}], "warnings"),
        ("solutions", [], "solver_metadata"),
    ],
)
def test_quality_audit_checks_first_repeat(field, value, check):
    result = valid_result()
    result["fit_details"]["optimized_gpu"][0][field] = value
    audit = logistic_wide.audit_quality(result)
    assert not audit["passed"]
    assert any(item["check"] == check and item["repeat"] == 0 for item in audit["failures"])


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("converged", False, "convergence"),
        ("stop_reason", "max_iter", "convergence"),
        ("grad_max_abs", 1.1e-4, "reported_gradient"),
        ("grad_max_abs", float("nan"), "reported_gradient"),
        ("objective", float("inf"), "reported_objective"),
    ],
)
def test_quality_audit_checks_every_solver_state(field, value, check):
    result = valid_result()
    result["fit_details"]["optimized_gpu"][0]["solver_calls"][0][field] = value
    audit = logistic_wide.audit_quality(result)
    assert not audit["passed"]
    assert any(item["check"] == check and item["repeat"] == 0 for item in audit["failures"])


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("gradient_max_abs_float64", 1.1e-4, "float64_gradient"),
        ("gradient_max_abs_float64", float("nan"), "float64_gradient"),
        ("objective_relative_gap_to_sklearn", 1.1e-5, "objective_accuracy"),
        ("objective_relative_gap_to_sklearn", float("nan"), "objective_accuracy"),
        ("objective_float64", float("inf"), "objective_accuracy"),
        ("prediction_agreement_sklearn", 0.994, "prediction_agreement"),
        ("prediction_agreement_sklearn", float("nan"), "prediction_agreement"),
    ],
)
def test_quality_audit_checks_independent_accuracy(field, value, check):
    result = valid_result()
    result["quality_last_fit"]["optimized_gpu"][field] = value
    audit = logistic_wide.audit_quality(result)
    assert not audit["passed"]
    assert any(item["check"] == check for item in audit["failures"])


def test_quality_audit_detects_missing_fit_metadata():
    result = valid_result()
    result["fit_details"]["optimized_gpu"].pop()
    audit = logistic_wide.audit_quality(result)
    assert not audit["passed"]
    assert audit["failures"][0]["check"] == "measured_fits"


def test_quality_audit_requires_actual_reference_native_execution():
    result = valid_result()
    result["parameters"]["require_reference_native_loop"] = True
    result["fit_details"]["previous_native"] = [
        {"implementation": "native_command_loop"}, {"implementation": "native_command_loop"}
    ]
    assert logistic_wide.audit_quality(result)["passed"]
    result["fit_details"]["previous_native"][0]["implementation"] = "native_metal"
    audit = logistic_wide.audit_quality(result)
    assert not audit["passed"]
    assert any(failure["check"] == "reference_native_loop" for failure in audit["failures"])


def test_frozen_native_snapshot_requires_hash_matched_extension(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {}, "native_binary_sha256": "0" * 64}))
    with pytest.raises(RuntimeError, match="requires --reference-extension"):
        with logistic_wide._loaded_reference(tmp_path, {}):
            pass
    binary = tmp_path / "_metal_loop.abi3.so"
    binary.write_bytes(b"different binary")
    with pytest.raises(RuntimeError, match="extension hash mismatch"):
        with logistic_wide._loaded_reference(tmp_path, {}, extension=binary):
            pass


def test_per_fit_independent_audit_checks_every_repeat_and_missing_scores():
    result = valid_result()
    result["parameters"]["audit_every_fit"] = True
    result["quality_each_fit"] = [copy.deepcopy(result["quality_last_fit"]["optimized_gpu"]) for _ in range(2)]
    audit = logistic_wide.audit_quality(result)
    assert audit["passed"]
    assert audit["independent_quality_fits_checked"] == 2
    result["quality_each_fit"][0]["gradient_max_abs_float64"] = 1.1e-4
    audit = logistic_wide.audit_quality(result)
    assert not audit["passed"]
    assert any(f["check"] == "float64_gradient" and f["repeat"] == 0 for f in audit["failures"])
    result["quality_each_fit"].pop()
    audit = logistic_wide.audit_quality(result)
    assert any(f["check"] == "independent_quality_fits" for f in audit["failures"])


def test_per_fit_quality_is_recorded_only_after_all_timing(monkeypatch):
    fit_metadata = valid_result()["fit_details"]["optimized_gpu"][0]
    score = valid_result()["quality_last_fit"]["optimized_gpu"]
    timed = []
    checked = []
    monkeypatch.setattr(logistic_wide, "make_data", lambda *args, **kwargs: ("X", "y", None))

    def fit(variant, *args, **kwargs):
        model = (variant, len(timed))
        timed.append(model)
        return 0.1, model, copy.deepcopy(fit_metadata)

    def quality(model, reference, *args):
        assert len(timed) == 12  # warmup plus all three repetitions and variants
        assert reference[0] == "sklearn"
        checked.append(model)
        return copy.deepcopy(score)

    monkeypatch.setattr(logistic_wide, "fit_once", fit)
    monkeypatch.setattr(logistic_wide, "quality", quality)
    result = logistic_wide.run_case(
        logistic_wide.Case("example", 10, 2), reference=None, seed=41, dtype="float32",
        warmup=1, repeats=3, max_iter=5, tol=1e-4, audit_every_fit=True,
    )
    assert result["quality_audit"]["passed"]
    assert result["quality_audit"]["independent_quality_fits_checked"] == 3
    fitted = [model for model in timed[3:] if model[0] == "optimized_gpu"]
    assert [model for model in checked if model[0] == "optimized_gpu"] == fitted[-1:] + fitted[:-1]


def test_quality_audit_requires_observed_native_loop_when_requested():
    result = valid_result()
    result["parameters"]["require_native_loop"] = True
    audit = logistic_wide.audit_quality(result)
    assert not audit["passed"]
    assert {failure["check"] for failure in audit["failures"]} == {"native_loop_required"}
    for fit in result["fit_details"]["optimized_gpu"]:
        fit["implementation"] = "native_command_loop"
        fit["native_loop_calls"] = ["completed"]
        fit["solver_calls"][-1]["implementation"] = "native_command_loop"
    assert logistic_wide.audit_quality(result)["passed"]
    result["fit_details"]["optimized_gpu"][0]["native_loop_calls"] = ["fallback"]
    audit = logistic_wide.audit_quality(result)
    assert not audit["passed"]
    assert any(failure["check"] == "gpu_implementation" for failure in audit["failures"])


@pytest.mark.parametrize("command_result", [None, object()])
def test_observed_implementation_records_completed_and_fallback_loops(command_result):
    state = SimpleNamespace(
        n_iter=2, converged=True, stop_reason="gtol", grad_max_abs=1e-5, objective=0.2
    )
    command = SimpleNamespace(run_native_loop=lambda *args: command_result)
    original_command = command.run_native_loop
    native = SimpleNamespace(native_command=command)

    def original_native():
        command.run_native_loop()
        return state

    native.minimize_binary = original_native
    estimator = SimpleNamespace(solve_logistic=lambda: state)
    solver = SimpleNamespace(minimize=lambda: state)
    with logistic_wide.observed_implementation(estimator, solver, native) as records:
        assert native.minimize_binary() is state
        expected = "fallback" if command_result is None else "completed"
        implementation = "native_metal" if command_result is None else "native_command_loop"
        assert records["native_loop_calls"] == [expected]
        assert records["solver_calls"][0]["implementation"] == implementation
    assert native.minimize_binary is original_native
    assert command.run_native_loop is original_command


def test_source_audit_detects_changed_current_and_frozen_sources(tmp_path):
    current = tmp_path / "current.py"
    frozen = tmp_path / "reference.py"
    current.write_text("current")
    frozen.write_text("frozen")
    payload = {
        "environment": {"source_sha256": {str(current): hashlib.sha256(b"current").hexdigest()}}
    }
    reference = SimpleNamespace(
        directory=tmp_path,
        manifest={"files": {"reference.py": hashlib.sha256(b"frozen").hexdigest()}},
        provenance={},
    )
    assert logistic_wide.audit_sources(payload, reference)["passed"]
    current.write_text("changed")
    frozen.unlink()
    audit = logistic_wide.audit_sources(payload, reference)
    assert not audit["passed"]
    assert set(audit["changed_or_missing"]) == {str(current), str(frozen)}


def test_source_audit_requires_and_validates_loaded_native_binary(monkeypatch, tmp_path):
    binary = tmp_path / "_metal_loop.abi3.so"
    binary.write_bytes(b"native-binary")
    module = types.ModuleType("_test_native_extension")
    module.__file__ = str(binary)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        logistic_wide.native_logistic,
        "native_command",
        SimpleNamespace(_metal_loop=module),
        raising=False,
    )
    result = valid_result()
    result["fit_details"]["optimized_gpu"][0]["implementation"] = "native_command_loop"
    payload = {
        "environment": {
            "source_sha256": {},
            "native_extension": logistic_wide.loaded_native_extension(),
        },
        "results": [result],
    }
    reference = SimpleNamespace(directory=tmp_path, manifest={"files": {}}, provenance={})
    assert logistic_wide.audit_sources(payload, reference)["passed"]
    binary.write_bytes(b"changed-binary")
    audit = logistic_wide.audit_sources(payload, reference)
    assert not audit["passed"]
    assert str(binary) in audit["changed_or_missing"]
    payload["environment"]["native_extension"] = None
    audit = logistic_wide.audit_sources(payload, reference)
    assert not audit["passed"]
    assert "native command loop completed without captured extension provenance" in audit[
        "changed_or_missing"
    ]


def test_audit_summary_keeps_failed_rows_and_requires_complete_source_audit():
    result = valid_result()
    result["quality_audit"] = logistic_wide.audit_quality(result)
    payload = {"results": [result]}
    logistic_wide.update_audit_summary(payload)
    assert payload["quality_audit"]["status"] == "incomplete"
    assert payload["quality_audit"]["passed"] is None
    logistic_wide.update_audit_summary(payload, complete=True)
    assert not payload["quality_audit"]["passed"]
    logistic_wide.update_audit_summary(payload, complete=True, source_audit={"passed": True})
    assert payload["quality_audit"]["passed"]
    result["quality_audit"]["passed"] = False
    logistic_wide.update_audit_summary(payload, complete=True, source_audit={"passed": True})
    assert not payload["quality_audit"]["passed"]
    assert payload["quality_audit"]["failed_cases"] == ["float32/test"]
    assert payload["results"] == [result]


def test_legacy_solver_adapter_drops_prepared_device_arguments_and_restores():
    calls = []

    def historical_solver(*args, **kwargs):
        calls.append((args, kwargs))
        return "historical-result"

    legacy = SimpleNamespace(solve_logistic=historical_solver, minimize=object())
    original = logistic_gpu.estimator_module.solve_logistic
    with logistic_gpu.selected_solver("legacy_mlx", legacy):
        result = logistic_gpu.estimator_module.solve_logistic(
            "X", "y", X_device=object(), X_center=object(), initial_parts=object(),
            certification_X=object(), C=1.0
        )
    assert result == "historical-result"
    assert calls == [(("X", "y"), {"C": 1.0})]
    assert logistic_gpu.estimator_module.solve_logistic is original
    assert legacy.solve_logistic is historical_solver


@pytest.mark.parametrize("body_raises", [False, True])
def test_profiler_times_context_entry_and_exit_separately(monkeypatch, body_raises):
    clock = [0.0]
    calls = []
    marker = object()

    @contextlib.contextmanager
    def preparation():
        clock[0] += 0.002
        calls.append("enter")
        try:
            yield marker
        except ValueError:
            calls.append("suppressed")
        finally:
            clock[0] += 0.003
            calls.append("exit")

    estimator = SimpleNamespace(_prepare_centered_logistic_input=preparation)
    solver = SimpleNamespace(_native_binary_objective=lambda: None)
    native = SimpleNamespace(minimize_binary=lambda: None)
    monkeypatch.setattr(logistic_profile.time, "perf_counter", lambda: clock[0])
    with logistic_profile.instrument(estimator, solver, native) as stages:
        with estimator._prepare_centered_logistic_input() as value:
            assert value is marker
            clock[0] += 17.0  # Expensive body is excluded from both context timings.
            if body_raises:
                raise ValueError("suppressed by original context")
    assert stages["centered_input_prepare_ms"] == pytest.approx(2.0)
    assert stages["centered_input_exit_drain_ms"] == pytest.approx(3.0)
    assert calls == (["enter", "suppressed", "exit"] if body_raises else ["enter", "exit"])
    assert estimator._prepare_centered_logistic_input is preparation


@pytest.mark.parametrize("quality_passed", [True, False])
def test_cli_persists_complete_report_before_quality_exit(tmp_path, monkeypatch, quality_passed):
    result = valid_result()
    result["case"].update(n_samples=20_000, n_features=1024)
    result["timings"] = {
        variant: {"samples_s": [0.1, 0.1], "median_ms": 100.0} for variant in logistic_wide.VARIANTS
    }
    result["speedup_over_previous_native"] = 1.0
    result["speedup_over_sklearn"] = 1.0
    if not quality_passed:
        result["fit_details"]["optimized_gpu"][0]["warnings"] = [{"message": "failed fit"}]
    result["quality_audit"] = logistic_wide.audit_quality(result)
    reference = SimpleNamespace(provenance={}, manifest={"files": {}}, directory=tmp_path)

    @contextlib.contextmanager
    def fake_reference(*args, **kwargs):
        yield reference

    monkeypatch.setattr(logistic_wide, "frozen_reference", fake_reference)
    monkeypatch.setattr(logistic_wide, "record_environment", lambda: {"source_sha256": {}})
    monkeypatch.setattr(logistic_wide, "run_case", lambda *args, **kwargs: result)
    monkeypatch.setattr(logistic_wide, "audit_sources", lambda *args: {"passed": True})
    output = tmp_path / "result.json"
    report = tmp_path / "report.md"
    exit_code = logistic_wide.main(
        [
            "--profile",
            "smoke",
            "--case",
            "wide_20k_1024",
            "--dtype",
            "float32",
            "--repeats",
            "3",
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )
    payload = json.loads(output.read_text())
    assert exit_code == (0 if quality_passed else 1)
    assert len(payload["results"]) == 1
    assert payload["quality_audit"]["passed"] is quality_passed
    assert f"Quality/source audit: {'PASS' if quality_passed else 'FAIL'}" in report.read_text()
    if not quality_passed:
        assert payload["results"][0]["quality_audit"]["failures"]
        assert "Failed audit for this row: warnings" in report.read_text()


@pytest.mark.parametrize(
    ("archive_name", "source_count", "kernel_count"),
    [
        ("native-logistic-2026-10-26.zip", 33, 6),
        ("wide-logistic-2026-10-26.zip", 35, 8),
        ("overhead-logistic-2026-10-26.zip", 37, 11),
    ],
)
def test_frozen_references_namespace_lazy_kernels_without_gpu_work(
    monkeypatch, archive_name, source_count, kernel_count
):
    names = []

    def fake_kernel(name, *args, **kwargs):
        names.append(name)
        return object()

    def forbidden_evaluation(*args, **kwargs):
        pytest.fail("Reference imports must not evaluate GPU work")

    monkeypatch.setattr(logistic_wide.mx.fast, "metal_kernel", fake_kernel)
    for function in ("eval", "async_eval", "synchronize"):
        monkeypatch.setattr(logistic_wide.mx, function, forbidden_evaluation)
    archive = logistic_wide.DEFAULT_REFERENCE_ARCHIVE.with_name(archive_name)
    with logistic_wide.frozen_reference(archive=archive) as reference:
        assert len(reference.manifest["files"]) == source_count
        assert len(names) == kernel_count
        assert names == reference.provenance["namespaced_kernels"]
        assert all(name.startswith(reference.provenance["kernel_name_prefix"]) for name in names)
        assert reference.provenance["kernel_name_prefix"] != logistic_wide.REFERENCE_KERNEL_PREFIX
        assert logistic_wide.mx.fast.metal_kernel is fake_kernel
        if archive_name.startswith("wide-"):
            assert "_solvers.native_logistic_fused" in reference.provenance["eager_kernel_modules"]
            assert "_kernels.validation" in reference.provenance["eager_kernel_modules"]
            assert "_mlxlearn_wide_reference._solvers.native_logistic_fused" in sys.modules
        source_directory = reference.directory
    assert not source_directory.exists()
    assert not any(name.startswith("_mlxlearn_wide_reference") for name in sys.modules)
    assert logistic_wide.mx.fast.metal_kernel is fake_kernel


@pytest.mark.parametrize(
    "archive_name", ("wide-logistic-2026-10-26.zip", "overhead-logistic-2026-10-26.zip")
)
def test_reference_matches_published_source_inventory(archive_name):
    archive = logistic_wide.DEFAULT_REFERENCE_ARCHIVE.with_name(archive_name)
    manifest = json.loads(archive.with_suffix(".manifest.json").read_text())
    result_path = archive.parents[2] / manifest["published_result"]
    published = json.loads(result_path.read_text())
    expected = {
        "mlxlearn/" + name.split("/src/mlxlearn/", 1)[1]: digest
        for name, digest in published["environment"]["source_sha256"].items()
        if "/src/mlxlearn/" in name
    }
    assert manifest["files"] == expected
    assert (
        hashlib.sha256(result_path.read_bytes()).hexdigest() == manifest["published_result_sha256"]
    )


@pytest.mark.parametrize("option", ["--reference-archive", "--reference-dir"])
def test_profiler_accepts_reference_override_without_gpu_work(monkeypatch, tmp_path, option):
    captured = []

    @contextlib.contextmanager
    def fake_reference(directory, *, archive):
        captured.append((directory, archive))
        yield SimpleNamespace(provenance={}, manifest={"files": {}}, directory=tmp_path)

    monkeypatch.setattr(logistic_profile, "frozen_reference", fake_reference)
    monkeypatch.setattr(logistic_profile, "CASES", ())
    monkeypatch.setattr(logistic_profile, "record_environment", lambda: {"source_sha256": {}})
    assert (
        logistic_profile.main([option, str(tmp_path), "--output", str(tmp_path / "profile.json")])
        == 0
    )
    expected = (None, Path(tmp_path)) if option.endswith("archive") else (Path(tmp_path), None)
    assert captured == [expected]


@pytest.mark.parametrize(
    ("attempt_status", "attempt_converged", "final_iter", "same_state", "outcome", "implementation"),
    [
        (1, True, 4, True, "completed", "native_command_loop"),
        (2, False, 4, True, "terminal_nonconvergence", "native_command_loop"),
        (0, False, 4, True, "terminal_nonconvergence", "native_command_loop"),
        (3, False, 4, True, "terminal_nonconvergence", "native_command_loop"),
        (2, False, 6, False, "resumed", "native_then_mlx"),
        (2, False, 4, False, "resumed_no_progress", "native_then_mlx"),
    ],
)
def test_observed_implementation_distinguishes_native_attempt_outcomes(
    attempt_status, attempt_converged, final_iter, same_state, outcome, implementation
):
    class NativeAttempt:
        def __init__(self, state, status):
            self.state, self.status = state, status

    attempt_state = SimpleNamespace(
        n_iter=4, converged=attempt_converged,
        stop_reason="gtol" if attempt_converged else "line-search-failure",
        grad_max_abs=1e-5, objective=0.2,
    )
    result = attempt_state if same_state else SimpleNamespace(
        n_iter=final_iter, converged=True, stop_reason="gtol", grad_max_abs=1e-5, objective=0.2,
    )
    command = SimpleNamespace(
        NativeAttempt=NativeAttempt,
        run_native_loop=lambda *args, **kwargs: NativeAttempt(attempt_state, attempt_status),
    )
    native = SimpleNamespace(native_command=command)

    def original_native():
        command.run_native_loop()
        return result

    native.minimize_binary = original_native
    estimator = SimpleNamespace(solve_logistic=lambda: result)
    solver = SimpleNamespace(minimize=lambda: result)
    with logistic_wide.observed_implementation(estimator, solver, native) as records:
        assert native.minimize_binary() is result
        assert records["native_loop_calls"] == [outcome]
        assert records["solver_calls"][0]["implementation"] == implementation
        assert records["native_loop_attempts"] == [{
            "status": attempt_status, "native_n_iter": 4, "final_n_iter": final_iter,
            "native_converged": attempt_converged, "outcome": outcome,
        }]


def test_quality_audit_allows_gpu_resume_unless_native_completion_required():
    result = valid_result()
    for fit in result["fit_details"]["optimized_gpu"]:
        fit["implementation"] = "native_then_mlx"
        fit["native_loop_calls"] = ["resumed"]
        fit["solver_calls"][-1]["implementation"] = "native_then_mlx"
    assert logistic_wide.audit_quality(result)["passed"]
    result["parameters"]["require_native_loop"] = True
    audit = logistic_wide.audit_quality(result)
    assert not audit["passed"]
    assert any(item["check"] == "native_loop_required" for item in audit["failures"])


def test_source_audit_requires_extension_provenance_for_native_resume(tmp_path):
    result = valid_result()
    result["fit_details"]["optimized_gpu"][0]["implementation"] = "native_then_mlx"
    payload = {"environment": {"source_sha256": {}, "native_extension": None}, "results": [result]}
    reference = SimpleNamespace(directory=tmp_path, manifest={"files": {}}, provenance={})
    audit = logistic_wide.audit_sources(payload, reference)
    assert not audit["passed"]
    assert "native command loop completed without captured extension provenance" in audit[
        "changed_or_missing"
    ]
