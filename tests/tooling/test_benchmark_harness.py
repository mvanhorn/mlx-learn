"""benchmarks/: the statistics, the gate's logic, the profiles, and one real run."""

from __future__ import annotations

import json

import numpy as np
import pytest
from benchmarks import gate, run
from benchmarks._harness import Comparison, _summarize, interquartile_range, measure

# --------------------------------------------------------------------- statistics


def test_one_outlier_does_not_set_the_spread():
    # The old formula returned max - min for n <= 7: 2.0 here.
    assert _summarize([1, 1, 1, 1, 1, 1, 3], "x").iqr_s == 0.0


@pytest.mark.parametrize("n", [2, 5, 7, 8, 11, 25])
def test_the_iqr_is_the_interpolated_quartile_range(n):
    samples = list(np.random.default_rng(n).lognormal(size=n))
    q1, q3 = np.percentile(samples, [25, 75])
    assert interquartile_range(samples) == pytest.approx(q3 - q1)


def test_measure_summarizes_like_compare():
    measurement = measure(lambda: None, label="noop", warmup=0, iterations=6)
    assert measurement.iterations == 6
    assert 0.0 <= measurement.iqr_s <= measurement.max_s - measurement.min_s
    # The cold-start measurement was documented but never wired up; the claim went
    # with the field rather than staying as a promise nothing kept.
    assert not hasattr(measurement, "cold_start_s")


def _comparison(mlx, sk):
    return Comparison(
        label="c", shape=(1, 1), mlxlearn=_summarize(mlx, "m"), sklearn=_summarize(sk, "s")
    )


def test_a_single_outlier_cannot_hide_a_2x_regression():
    rng = np.random.default_rng(0)
    sk = list(1.0 + 0.01 * rng.normal(size=7))
    sk[3] = 5.0  # one garbage-collection pause on the scikit-learn side
    mlx = list(2.0 + 0.01 * rng.normal(size=7))
    comparison = _comparison(mlx, sk)
    assert comparison.speedup < 0.55
    assert comparison.significant


def test_equal_distributions_are_within_noise():
    rng = np.random.default_rng(1)
    comparison = _comparison(list(rng.lognormal(0, 0.1, 7)), list(rng.lognormal(0, 0.1, 7)))
    assert not comparison.significant


def test_too_few_samples_never_claim_significance():
    assert not _comparison([2.0, 2.01, 2.02, 2.03], [1.0, 1.01, 1.02, 1.03]).significant


# --------------------------------------------------------------------------- gate


def _result(label="knn/kneighbors/4000x32", *, speedup, significant=True, n=4000, **extra):
    algorithm, operation = label.split("/")[:2]
    return {
        "label": label,
        "algorithm": algorithm,
        "operation": operation,
        "shape": [n, 32],
        "speedup": speedup,
        "speedup_best": speedup,
        "significant": significant,
        **extra,
    }


def test_a_self_derived_crossover_no_longer_excuses_the_mlx_path():
    # MLX regressed at 500-8000 rows, which moved the crossover measured in the same
    # run up to 32000 — the old gate then called every regression "by design".
    results = [
        _result(f"knn/kneighbors/{n}x32", speedup=0.45, n=n, backend="mlx")
        for n in (500, 2_000, 8_000)
    ] + [_result("knn/kneighbors/32000x32", speedup=3.0, n=32_000, backend="mlx")]
    payload = {
        "layer": "patched",
        "results": results,
        "crossovers": {"knn/kneighbors": {"crossover_n_samples": 32_000}},
    }
    failures, notes = gate.evaluate(payload)
    assert len(failures) == 3
    assert all("Tuning" in failure for failure in failures)
    assert not any("by design" in note for note in notes)
    assert any("measured crossover 32000" in note for note in notes)


def test_patched_handoff_overhead_is_gated():
    payload = {
        "layer": "patched",
        "results": [_result("logreg/fit/5000x32", speedup=0.6, backend="sklearn")],
    }
    failures, _ = gate.evaluate(payload)
    assert len(failures) == 1 and "overhead" in failures[0]


def test_the_layer1_cpu_path_is_not_the_claim():
    payload = {
        "layer": "direct",
        "results": [_result("svc/fit/1000x32", speedup=0.2, backend="cpu")],
    }
    assert gate.evaluate(payload)[0] == []


def test_forced_kernels_below_the_threshold_are_reported_not_gated():
    payload = {
        "layer": "direct",
        "force_backend": "mlx",
        "results": [
            _result("logreg/fit/1000x16", speedup=0.03, backend="mlx", dispatch="sklearn"),
            _result("svc/fit/1000x512", speedup=2.0, backend="mlx", dispatch="sklearn"),
        ],
    }
    failures, notes = gate.evaluate(payload)
    assert failures == []
    assert any("too conservative" in note for note in notes)


def test_forced_kernels_the_threshold_selects_are_gated():
    payload = {
        "layer": "direct",
        "force_backend": "mlx",
        "results": [_result("logreg/fit/60000x1024", speedup=0.5, backend="mlx", dispatch="mlx")],
    }
    assert len(gate.evaluate(payload)[0]) == 1


def test_accepted_regressions_and_noise_pass():
    payload = {
        "results": [
            _result("knn/fit/4000x32", speedup=0.2, backend="mlx"),
            _result("svc/fit/4000x256", speedup=0.7, significant=False, backend="mlx"),
            _result("svc/fit/8000x256", speedup=0.97, backend="mlx"),  # inside TOLERANCE
        ]
    }
    assert gate.evaluate(payload)[0] == []


def test_a_schema_1_payload_without_backends_is_gated():
    payload = {"results": [_result(speedup=0.5)], "crossovers": {}}
    assert len(gate.evaluate(payload)[0]) == 1


# ----------------------------------------------------------------------- profiles


@pytest.mark.parametrize("profile", sorted(run.PROFILES))
def test_case_labels_are_unique(profile):
    labels = [case.label for case in run.PROFILES[profile]]
    assert len(labels) == len(set(labels))


def test_the_stock_baseline_is_the_stock_default():
    _, make_sk = run._estimators(run.Case("knn", "fit", 10, 3), "direct")
    assert make_sk().algorithm == "auto"
    _, make_sk = run._estimators(
        run.Case("knn", "fit", 10, 3, options={"sklearn_algorithm": "brute"}), "direct"
    )
    assert make_sk().algorithm == "brute"


def test_the_full_grid_covers_the_regimes_found_missing():
    full = run.PROFILES["full"]
    knn_queries = {(c.n_features, c.n_queries) for c in full if c.operation == "kneighbors"}
    for d in (2, 3, 8):
        assert (d, 256) in knn_queries
    for d in (8, 32):
        assert {(d, 10_000), (d, 50_000)} <= knn_queries
    classes = {c.option("n_classes") for c in full if c.algorithm == "svc"}
    assert {8, 32, 128} <= classes


def test_the_published_profile_encodes_the_documented_headline_cases():
    labels = {case.label for case in run.PROFILES["published"]}
    assert "knn/kneighbors/128000x768/q256/sklearn_algorithm=brute" in labels
    assert "svc/fit/4000x256" in labels
    assert "svc/fit/2500x20/C=10.0/kernel=linear/n_classes=4" in labels
    assert "logreg/fit/60000x1024" in labels
    assert run.PROFILE_SETTINGS["published"] == {
        "layer": "direct",
        "force_backend": "mlx",
        "iterations": 25,
    }


def test_smoke_reaches_past_every_mlx_threshold():
    from mlxlearn._common.config import get_tuning

    tuning = get_tuning()
    smoke = run.PROFILES["smoke"]
    assert any(
        c.algorithm == "logreg"
        and c.n_samples >= tuning.logreg_min_samples
        and c.n_samples * c.n_features >= tuning.logreg_min_work
        for c in smoke
    )
    assert any(
        c.algorithm == "svc"
        and c.n_samples >= tuning.svc_min_samples
        and c.n_samples * c.n_features >= tuning.svc_min_work
        for c in smoke
    )


# --------------------------------------------------------------------- end to end

_TINY = [
    run.Case("knn", "kneighbors", 200, 8, 32),
    run.Case("logreg", "fit", 300, 4),
    run.Case("svc", "fit", 240, 4, options={"n_classes": 3}),
]


def test_a_run_records_backends_and_the_gate_reads_them(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(run.PROFILES, "tiny", _TINY)
    output, report = tmp_path / "results.json", tmp_path / "report.md"
    args = ["--profile", "tiny", "--iterations", "1", "--output", str(output)]
    assert run.main([*args, "--report", str(report)]) == 0

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 2 and payload["layer"] == "patched"
    backends = {r["label"]: r["backend"] for r in payload["results"]}
    assert set(backends) == {case.label for case in _TINY}
    # Every one of these is far below its threshold: patched dispatch hands it off.
    assert set(backends.values()) == {"sklearn"}
    assert all(r["dispatch"] == r["backend"] for r in payload["results"])
    text = report.read_text()
    assert "interquartile range" in text and "three standard errors" in text

    assert gate.main(["--results", str(output)]) == 0


@pytest.mark.gpu
def test_a_forced_run_records_what_dispatch_would_have_done(tmp_path, monkeypatch):
    monkeypatch.setitem(run.PROFILES, "tiny", _TINY[:1])
    output = tmp_path / "results.json"
    argv = ["--profile", "tiny", "--iterations", "1", "--output", str(output)]
    assert run.main([*argv, "--layer", "direct", "--force-backend", "mlx"]) == 0
    (result,) = json.loads(output.read_text())["results"]
    assert result["backend"] == "mlx" and result["force_backend"] == "mlx"
    assert result["dispatch"] == "sklearn"
