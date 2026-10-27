"""Fast label encoding retains NumPy's sorted class mapping."""

import numpy as np
import pytest

from mlxlearn.exceptions import UnsupportedInputError
from mlxlearn.linear_model._logistic import (
    MLXLogisticMixin,
    _encode_labels,
    _prepare_logistic_input,
)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([7, -11, 7, -11], dtype=np.int32),
        np.array([2**64 - 1, 2**63, 2**64 - 1], dtype=np.uint64),
        np.array([True, False, True]),
        np.array([False, False]),
        np.array([3, 3, 3]),
        np.array([8, -2, 3, -2, 8]),
        np.array([3.0, -1.0, 3.0]),
        np.array(["zebra", "ant", "zebra"]),
        np.array(["zebra", "ant", "fox"], dtype=object),
    ],
)
def test_label_mapping_matches_sorted_unique(labels):
    before = labels.copy()
    expected_classes, expected_codes = np.unique(labels, return_inverse=True)
    classes, codes = _encode_labels(labels)
    np.testing.assert_array_equal(classes, expected_classes)
    np.testing.assert_array_equal(codes, expected_codes)
    np.testing.assert_array_equal(labels, before)
    assert classes.dtype == labels.dtype
    assert codes.dtype == np.int32


def test_checked_cast_reuses_conversion_without_changing_input():
    X = np.arange(120.0).reshape(12, 10)[:, ::2]
    original = X.copy()
    prepared = _prepare_logistic_input(X)
    np.testing.assert_array_equal(prepared, X.astype(np.float32))
    np.testing.assert_array_equal(X, original)
    assert prepared.dtype == np.float32
    assert prepared.flags.c_contiguous
    assert not np.shares_memory(prepared, X)


@pytest.mark.parametrize("other", [0.0, np.nan, np.inf, -np.inf])
def test_checked_cast_preserves_finite_overflow_capability_error(other):
    with pytest.raises(UnsupportedInputError, match="infinite in float32"):
        _prepare_logistic_input(np.array([[1e300, other]]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_checked_cast_leaves_invalid_nonfinite_values_for_validation(bad):
    prepared = _prepare_logistic_input(np.array([[bad, 1.0]]))
    np.testing.assert_array_equal(prepared, np.array([[bad, 1.0]], dtype=np.float32))


def test_checked_cast_preserves_array_subclass_validation():
    # validate_data must still see the original subclass and enforce its rules.
    class ArraySubclass(np.ndarray):
        pass

    X = np.arange(4.0).reshape(2, 2).view(ArraySubclass)
    assert _prepare_logistic_input(X) is X


@pytest.mark.parametrize(
    "rows, columns, below",
    [(5000, 32, True), (20000, 512, True), (20000, 1024, False),
     (249999, 32, True), (250000, 32, False), (500000, 16, False),
     (250000, 16, True)],
)
def test_tall_crossover_keeps_both_size_floors(rows, columns, below):
    assert MLXLogisticMixin()._below_crossover(rows, columns)[0] is below
