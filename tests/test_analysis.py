"""Tests for reconstruction analysis and validation metrics."""

import numpy as np
import pytest

from thermoreconlab.analysis import (
    compute_all_metrics,
    compute_error_field,
    mae,
    max_absolute_error,
    relative_l2_error,
    residual_norm,
    rmse,
    validate_reconstruction,
)
from thermoreconlab.exceptions import ValidationError


def test_rmse_is_correct() -> None:
    """RMSE should match its mathematical definition."""
    true_values = np.array([0.0, 1.0, 2.0])
    predicted_values = np.array([0.0, 2.0, 4.0])

    result = rmse(true_values, predicted_values)

    assert result == pytest.approx(np.sqrt(5.0 / 3.0))


def test_mae_is_correct() -> None:
    """MAE should average the absolute errors."""
    true_values = np.array([0.0, 1.0, 2.0])
    predicted_values = np.array([0.0, 2.0, 4.0])

    result = mae(true_values, predicted_values)

    assert result == pytest.approx(1.0)


def test_relative_l2_error_is_correct() -> None:
    """Relative L2 error should use the true-value norm."""
    true_values = np.array([0.0, 1.0, 2.0])
    predicted_values = np.array([0.0, 2.0, 4.0])

    result = relative_l2_error(
        true_values,
        predicted_values,
    )

    assert result == pytest.approx(1.0)


def test_max_absolute_error_is_correct() -> None:
    """The maximum pointwise error should be returned."""
    true_values = np.array([0.0, 1.0, 2.0])
    predicted_values = np.array([0.0, 2.0, 4.0])

    result = max_absolute_error(
        true_values,
        predicted_values,
    )

    assert result == pytest.approx(2.0)


def test_residual_norm_is_correct() -> None:
    """Residual norm should compare predicted and observed data."""
    predicted = np.array([1.0, 2.0, 3.0])
    observed = np.array([1.0, 1.0, 1.0])

    result = residual_norm(predicted, observed)

    assert result == pytest.approx(np.sqrt(5.0))


def test_identical_arrays_have_zero_errors() -> None:
    """All error metrics should vanish for identical arrays."""
    values = np.arange(9, dtype=float).reshape(3, 3)

    assert rmse(values, values) == pytest.approx(0.0)
    assert mae(values, values) == pytest.approx(0.0)
    assert relative_l2_error(
        values,
        values,
    ) == pytest.approx(0.0)
    assert max_absolute_error(
        values,
        values,
    ) == pytest.approx(0.0)


def test_zero_truth_and_zero_prediction_have_zero_relative_error() -> None:
    """Two zero arrays should have zero relative error."""
    true_values = np.zeros((3, 3))
    predicted_values = np.zeros((3, 3))

    result = relative_l2_error(
        true_values,
        predicted_values,
    )

    assert result == pytest.approx(0.0)


def test_zero_truth_and_nonzero_prediction_have_infinite_error() -> None:
    """A nonzero estimate relative to zero truth is undefined."""
    true_values = np.zeros((3, 3))
    predicted_values = np.ones((3, 3))

    result = relative_l2_error(
        true_values,
        predicted_values,
    )

    assert np.isinf(result)


def test_error_field_has_correct_values() -> None:
    """The error field should be reconstructed minus true."""
    true_source = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    reconstructed_source = np.array(
        [
            [1.5, 1.0],
            [4.0, 4.0],
        ]
    )

    error = compute_error_field(
        true_source,
        reconstructed_source,
    )

    expected = np.array(
        [
            [0.5, -1.0],
            [1.0, 0.0],
        ]
    )

    assert np.array_equal(error, expected)


def test_compute_all_metrics_returns_expected_keys() -> None:
    """The metric summary should contain all standard metrics."""
    true_source = np.ones((3, 3))
    reconstructed_source = np.zeros((3, 3))

    metrics = compute_all_metrics(
        true_source,
        reconstructed_source,
    )

    assert set(metrics) == {
        "rmse",
        "mae",
        "relative_l2_error",
        "max_absolute_error",
    }

    assert all(
        isinstance(value, float)
        for value in metrics.values()
    )


def test_validate_reconstruction_matches_metric_summary() -> None:
    """Validation should use the standard metric collection."""
    true_source = np.ones((3, 3))
    reconstructed_source = np.zeros((3, 3))

    expected = compute_all_metrics(
        true_source,
        reconstructed_source,
    )
    result = validate_reconstruction(
        true_source,
        reconstructed_source,
    )

    assert result == expected


def test_metrics_reject_mismatched_shapes() -> None:
    """Metric inputs must have matching shapes."""
    true_values = np.ones((3, 3))
    predicted_values = np.ones((3, 4))

    with pytest.raises(ValidationError):
        rmse(true_values, predicted_values)


def test_error_field_rejects_non_2d_input() -> None:
    """Error fields must be constructed from 2D arrays."""
    with pytest.raises(ValidationError):
        compute_error_field(
            np.ones(5),
            np.ones(5),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_metrics_reject_non_finite_values(
    invalid_value: float,
) -> None:
    """Metrics should reject NaN and infinite input values."""
    true_values = np.ones((3, 3))
    predicted_values = np.ones((3, 3))
    predicted_values[1, 1] = invalid_value

    with pytest.raises(ValidationError):
        mae(true_values, predicted_values)