"""Tests for reconstruction analysis and validation metrics."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike

from thermoreconlab.analysis import (
    compute_all_metrics,
    compute_error_field,
    mae,
    max_absolute_error,
    relative_l2_error,
    relative_residual,
    residual_norm,
    residual_rms,
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


def test_relative_residual_is_correct() -> None:
    """Relative residual should use the observed-data norm."""
    predicted = np.array([2.0, 2.0])
    observed = np.array([1.0, 1.0])

    result = relative_residual(predicted, observed)

    assert result == pytest.approx(1.0)


def test_residual_rms_is_correct() -> None:
    """Residual RMS should use the number of measurements."""
    predicted = np.array([1.0, 3.0, 5.0])
    observed = np.array([1.0, 1.0, 1.0])

    result = residual_rms(predicted, observed)

    assert result == pytest.approx(np.sqrt(20.0 / 3.0))


def test_normalized_residuals_are_zero_for_identical_data() -> None:
    """Identical measurements should produce zero residuals."""
    values = np.array([1.0, 2.0, 3.0])

    assert relative_residual(values, values) == pytest.approx(0.0)
    assert residual_rms(values, values) == pytest.approx(0.0)


def test_relative_residual_zero_observed_and_zero_residual() -> None:
    """Zero observations and zero residual should return zero."""
    zeros = np.zeros(3)

    assert relative_residual(zeros, zeros) == pytest.approx(0.0)


def test_relative_residual_zero_observed_and_nonzero_residual() -> None:
    """Nonzero residual relative to zero observations is infinite."""
    predicted = np.ones(3)
    observed = np.zeros(3)

    assert np.isinf(relative_residual(predicted, observed))


@pytest.mark.parametrize(
    "metric",
    [residual_norm, relative_residual, residual_rms],
)
def test_residual_metrics_reject_mismatched_shapes(
    metric: Callable[[ArrayLike, ArrayLike], float],
) -> None:
    """Residual metrics should reject mismatched shapes."""
    with pytest.raises(ValidationError):
        metric(np.ones(3), np.ones(4))


@pytest.mark.parametrize(
    "metric",
    [residual_norm, relative_residual, residual_rms],
)
def test_residual_metrics_reject_empty_input(
    metric: Callable[[ArrayLike, ArrayLike], float],
) -> None:
    """Residual metrics should reject empty arrays."""
    with pytest.raises(ValidationError):
        metric(np.array([]), np.array([]))


@pytest.mark.parametrize(
    "metric",
    [residual_norm, relative_residual, residual_rms],
)
def test_residual_metrics_reject_non_finite_input(
    metric: Callable[[ArrayLike, ArrayLike], float],
) -> None:
    """Residual metrics should reject non-finite values."""
    predicted = np.array([1.0, np.nan])
    observed = np.array([1.0, 2.0])

    with pytest.raises(ValidationError):
        metric(predicted, observed)


@pytest.mark.parametrize(
    "metric",
    [residual_norm, relative_residual, residual_rms],
)
def test_residual_metrics_require_one_dimensional_input(
    metric: Callable[[ArrayLike, ArrayLike], float],
) -> None:
    """Residual metrics should require one-dimensional arrays."""
    values = np.ones((2, 2))

    with pytest.raises(ValidationError):
        metric(values, values)


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


def test_error_field_uses_only_interior_source_nodes() -> None:
    """Boundary source differences should not enter the error field."""
    true_source = np.zeros((4, 4))
    reconstructed_source = np.full((4, 4), 100.0)
    reconstructed_source[1:-1, 1:-1] = np.array(
        [
            [1.0, -2.0],
            [3.0, 0.0],
        ]
    )

    error = compute_error_field(
        true_source,
        reconstructed_source,
    )

    expected = np.zeros((4, 4))
    expected[1:-1, 1:-1] = np.array(
        [
            [1.0, -2.0],
            [3.0, 0.0],
        ]
    )

    assert np.array_equal(error, expected)


def test_source_metrics_ignore_boundary_values() -> None:
    """Source metrics should exclude non-reconstructed boundaries."""
    true_source = np.ones((5, 6))
    reconstructed_source = true_source.copy()

    reconstructed_source[0, :] = 100.0
    reconstructed_source[-1, :] = -100.0
    reconstructed_source[:, 0] = 50.0
    reconstructed_source[:, -1] = -50.0

    metrics = compute_all_metrics(
        true_source,
        reconstructed_source,
    )

    assert all(
        value == pytest.approx(0.0)
        for value in metrics.values()
    )


def test_source_metrics_detect_interior_difference() -> None:
    """An interior source difference should produce nonzero errors."""
    true_source = np.ones((5, 5))
    reconstructed_source = true_source.copy()
    reconstructed_source[2, 2] = 3.0

    metrics = compute_all_metrics(
        true_source,
        reconstructed_source,
    )

    assert metrics["rmse"] > 0.0
    assert metrics["mae"] > 0.0
    assert metrics["relative_l2_error"] > 0.0
    assert metrics["max_absolute_error"] == pytest.approx(2.0)


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