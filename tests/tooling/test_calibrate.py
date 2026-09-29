"""benchmarks/calibrate.py: the grids behind the thresholds, and their tables."""

from __future__ import annotations

import json
import math
import re

import pytest
from benchmarks import calibrate
from benchmarks.calibrate import GRIDS, Point

from mlxlearn._common.config import get_tuning

from ._load import REPO


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_every_measurement_has_its_own_key(name):
    keys = [key for point in GRIDS[name].points for key in point.keys()]
    assert len(keys) == len(set(keys))


def test_the_dense_neighbor_grid_times_every_batch_size_on_both_sides_of_the_floors():
    """A fit cannot see how it will be queried, so the grid behind the neighbor floors
    times a dozen batch sizes on every shape, from below the work floor to well above
    it, at rows that are not powers of two."""
    tuning = get_tuning()
    points = GRIDS["knn"].points
    assert {p.d for p in points} == {16, 24, 32, 48, 64, 96, 128, 256}
    for d in {p.d for p in points}:
        column = [p for p in points if p.d == d]
        assert all({1, 2, 4, 8, 16, 32, 64, 128, 256, 512} <= set(p.queries) for p in column)
        if d < tuning.knn_wide_features:
            rows = [p.n for p in column]
            assert min(rows) < tuning.knn_narrow_min_samples <= max(rows)
        else:
            works = [p.n * p.d for p in column]
            assert min(works) < tuning.knn_min_work <= max(works)
        assert any(p.n & (p.n - 1) for p in column)


def _rows(p):
    return p.n


def _work(p):
    return p.n * p.d


def _width(p):
    return p.d


def _pair_work(p):
    return 2 * p.n / p.n_classes * p.d


@pytest.mark.parametrize(
    "names, knob, size",
    [
        (("knn-rows",), "knn_min_train_samples", _rows),
        (("knn-rows",), "knn_min_work", _work),
        (("knn-tree", "knn", "knn-narrow"), "knn_min_features", _width),
        (("knn",), "knn_wide_features", _width),
        (("knn",), "knn_narrow_min_samples", _rows),
        (("svc",), "svc_min_features", _width),
        (("svc",), "svc_min_samples", _rows),
        (("svc",), "svc_min_work", _work),
        (("svc-multi", "svc-multi-overlap"), "svc_min_pair_work", _pair_work),
        (("svc-params",), "svc_min_gamma_ratio", lambda p: p.gamma or 1.0),
        # A ceiling: C at most the knob stays on MLX, so C = 1 must sort below it.
        (("svc-params",), "svc_low_gamma_max_c", lambda p: math.nextafter(p.C, 0.0)),
        (("svc-params",), "svc_poly_min_degree", lambda p: p.degree),
        (("logreg-multi",), "logreg_min_samples", _rows),
        (("logreg-binary", "logreg-labels"), "logreg_binary_min_samples", _rows),
        (("logreg-binary", "logreg-labels"), "logreg_binary_max_features", lambda p: p.d - 1),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_threshold_has_a_grid_on_both_sides_of_it(names, knob, size):
    """A threshold is only as good as the measurements either side of it."""
    points = [p for name in names for p in GRIDS[name].points]
    floor = getattr(get_tuning(), knob)
    assert min(map(size, points)) < floor <= max(map(size, points))


def _committed(grid):
    results = sorted((REPO / "benchmarks" / "results" / "calibration").glob(f"{grid}-*.jsonl"))
    return calibrate.load(results, grid)


def test_the_data_probe_threshold_has_measurements_on_both_sides_of_it():
    """The data rule's knobs are read off what the probe found, so the committed
    results must hold data it passed and data it did not, on both versions, and
    integer data on both sides of the width it keeps on MLX from."""
    tuning = get_tuning()
    records = _committed("knn-data")
    for version in ("1.9.0", "1.7.2"):
        mine = [r for r in records if r["sklearn"] == version]
        fractions = {r["probe_uncertified"] / r["probe_rows"] for r in mine if r["probe_rows"]}
        assert min(fractions) <= tuning.knn_probe_max_uncertified < max(fractions)
        widths = {r["d"] for r in mine if r["integer_path"]}
        assert min(widths) < tuning.knn_wide_features <= max(widths)


def test_the_host_inference_threshold_has_measurements_on_both_sides_of_it():
    tuning = get_tuning()
    for version in ("1.9.0", "1.7.2"):
        works = [
            r["q"] * r["n_sv"] * r["d"] for r in _committed("svc-predict")
            if r["sklearn"] == version
        ]
        assert min(works) <= tuning.svc_host_inference_work < max(works)


def _record(grid, n, d, speedup, significant=True, version="1.9.0", **extra):
    point = {"knn": "knn", "svc": "svc"}.get(grid.split("-")[0], "logreg")
    base = {
        "grid": grid, "estimator": point, "n": n, "d": d, "algorithm": "auto",
        "data": "gauss", "kernel": "rbf", "C": 1.0, "n_classes": 2, "sklearn": version,
        "operation": "fit", "speedup": speedup, "significant": significant,
    }
    return {**base, **extra}


def test_a_table_marks_losses_and_noise_and_says_where_dispatch_sends_each_row():
    records = [
        _record("svc", 4_000, 16, 0.50, kernel="poly", C=10.0),
        _record("svc", 4_000, 16, 0.60, version="1.7.2", kernel="poly", C=10.0),
        _record("svc", 4_000, 256, 2.00, kernel="poly", C=10.0),
        _record("svc", 4_000, 256, 1.02, significant=False, version="1.7.2",
                kernel="poly", C=10.0),
    ]
    text = calibrate.render("svc", records)
    lines = text.splitlines()
    assert "1.9.0 / 1.7.2" in lines[0]
    assert lines[1] == "| rows × features | poly, C = 10 | dispatch |"
    assert "| 4 000 × 16 | **0.50** / **0.60** | scikit-learn |" in lines
    assert "| 4 000 × 256 | 2.00 / 1.02~ | MLX |" in lines


def test_the_dense_neighbor_table_keeps_the_worse_version():
    records = [
        _record("knn", 2_000_000, 64, 1.30, operation="kneighbors", q=16),
        _record("knn", 2_000_000, 64, 0.80, operation="kneighbors", q=16, version="1.7.2"),
        _record("knn", 2_000_000, 64, 1.50, operation="kneighbors", q=256),
        _record("knn", 2_000_000, 64, 1.40, operation="kneighbors", q=256, version="1.7.2"),
    ]
    lines = calibrate.render("knn", records).splitlines()
    assert lines[1] == "| rows × features | 16 queries | 256 queries | dispatch |"
    assert lines[3] == "| 2 000 000 × 64 | **0.80** | 1.40 | MLX |"


def test_a_row_dispatched_both_ways_names_the_columns_sent_to_mlx():
    records = [
        _record("logreg-binary", 200_000, 16, 1.8),
        _record("logreg-binary", 200_000, 256, 0.8),
    ]
    lines = calibrate.render("logreg-binary", records).splitlines()
    assert lines[-1] == "| 200 000 | 1.80 | **0.80** | MLX at 16 features |"


def test_update_re_renders_the_tables_a_document_holds_and_nothing_else():
    records = [_record("logreg-binary", 200_000, 16, 1.8)]
    stale = calibrate.render("logreg-binary", [_record("logreg-binary", 200_000, 16, 0.5)])
    text = f"# Title\n\nProse | with a pipe.\n\n{stale}\nMore prose.\n"
    updated, count = calibrate.update_document(text, records)
    assert count == 1
    assert updated == text.replace(stale, calibrate.render("logreg-binary", records))
    assert "| 200 000 | 1.80 | MLX |" in updated
    # Results for another grid leave this table as it was.
    other = [_record("svc", 4_000, 256, 2.0)]
    assert calibrate.update_document(text, other) == (text, 0)


def test_the_committed_results_hold_every_measurement_of_every_grid():
    results = sorted((REPO / "benchmarks" / "results" / "calibration").glob("*.jsonl"))
    measured = {record["key"] for record in calibrate.load(results)}
    for name, grid in GRIDS.items():
        missing = [key for point in grid.points for key in point.keys() if key not in measured]
        assert not missing, f"grid {name!r} has no committed result for {missing[:3]}"


def test_the_committed_results_render_the_published_tables():
    """docs/benchmarks.md's calibration tables are what the committed results render."""
    results = sorted((REPO / "benchmarks" / "results" / "calibration").glob("*.jsonl"))
    assert results
    text = (REPO / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    updated, count = calibrate.update_document(text, calibrate.load(results))
    assert count >= 8
    assert updated == text, "docs/benchmarks.md is stale: run `benchmarks.calibrate update`"


def test_a_run_writes_records_and_resumes_where_it_stopped(tmp_path, monkeypatch):
    tiny = calibrate.Grid(
        "tiny",
        (
            Point("knn", 300, 20, (1, 8), fit=True),
            Point("svc", 120, 6),
            Point("logreg", 150, 4, n_classes=3, data="blobs"),
        ),
        layout="knn-q",
        iterations=1,
    )
    monkeypatch.setitem(GRIDS, "tiny", tiny)
    output = tmp_path / "tiny.jsonl"
    lines = []
    assert calibrate.run("tiny", output, limit=2, echo=lines.append) == 2
    assert calibrate.run("tiny", output, echo=lines.append) == 3
    assert calibrate.run("tiny", output, echo=lines.append) == 0
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["key"] for r in records] == [k for p in tiny.points for k in p.keys()]
    assert all(r["sklearn"] and r["grid"] == "tiny" for r in records)
    assert {r["operation"] for r in records} == {"fit", "kneighbors"}
    assert len(lines) == 5


def test_every_calibration_table_in_the_docs_names_a_grid():
    """docs/benchmarks.md cites its calibration tables by grid; each one must exist."""
    text = (REPO / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    cited = set(re.findall(r"python -m benchmarks\.calibrate table ([\w-]+)", text))
    assert cited, "docs/benchmarks.md cites no calibration grid"
    assert cited <= set(GRIDS)
    # Every grid that calibrates a threshold is on the record.
    assert {"knn", "knn-rows", "knn-tree", "knn-narrow", "knn-data", "knn-k", "svc",
            "svc-sigmoid", "svc-multi", "svc-multi-overlap", "svc-params", "svc-predict",
            "logreg-binary", "logreg-multi", "logreg-separable", "logreg-labels"} <= cited
