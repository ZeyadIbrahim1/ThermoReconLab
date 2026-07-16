"""Numerical analysis and validation metrics for ThermoReconLab."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from thermoreconlab.core.fields import ensure_2d_array
from thermoreconlab.exceptions import ValidationError


def _as_finite_array(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Convert input values to a finite floating-point array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"{name} must contain numeric values."
        ) from error

    if array.ndim == 0:
        raise ValidationError(
            f"{name} must contain at least one dimension."
        )

    if array.size == 0:
        raise ValidationError(f"{name} must not be empty.")

    if not np.all(np.isfinite(array)):
        raise ValidationError(
            f"{name} must contain only finite values."
        )

    return array.copy()


def _validate_matching_arrays(
    reference: ArrayLike,
    estimate: ArrayLike,
    *,
    reference_name: str = "reference",
    estimate_name: str = "estimate",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate two numeric arrays with matching shapes."""
    reference_array = _as_finite_array(
        reference,
        name=reference_name,
    )
    estimate_array = _as_finite_array(
        estimate,
        name=estimate_name,
    )

    if reference_array.shape != estimate_array.shape:
        raise ValidationError(
            f"{reference_name} and {estimate_name} must have "
            f"matching shapes, but received "
            f"{reference_array.shape} and {estimate_array.shape}."
        )

    return reference_array, estimate_array


def _validate_source_fields(
    true_source: ArrayLike,
    reconstructed_source: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate matching two-dimensional source fields."""
    true_array = ensure_2d_array(
        true_source,
        name="true_source",
    )
    reconstructed_array = ensure_2d_array(
        reconstructed_source,
        name="reconstructed_source",
    )

    if true_array.shape != reconstructed_array.shape:
        raise ValidationError(
            "true_source and reconstructed_source must have "
            f"matching shapes, but received {true_array.shape} "
            f"and {reconstructed_array.shape}."
        )

    if true_array.shape[0] < 3 or true_array.shape[1] < 3:
        raise ValidationError(
            "Source fields must have at least three grid points "
            "in each direction."
        )

    return true_array, reconstructed_array


def _source_interiors(
    true_source: ArrayLike,
    reconstructed_source: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return matching interior source values for evaluation."""
    true_array, reconstructed_array = _validate_source_fields(
        true_source,
        reconstructed_source,
    )

    return (
        true_array[1:-1, 1:-1],
        reconstructed_array[1:-1, 1:-1],
    )


def _validate_measurement_arrays(
    predicted_measurements: ArrayLike,
    observed_measurements: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate matching one-dimensional measurement arrays."""
    predicted_array, observed_array = _validate_matching_arrays(
        predicted_measurements,
        observed_measurements,
        reference_name="predicted_measurements",
        estimate_name="observed_measurements",
    )

    if predicted_array.ndim != 1:
        raise ValidationError(
            "predicted_measurements and observed_measurements "
            "must be one-dimensional."
        )

    return predicted_array, observed_array


def rmse(
    true_values: ArrayLike,
    predicted_values: ArrayLike,
) -> float:
    """Return the root mean squared error."""
    true_array, predicted_array = _validate_matching_arrays(
        true_values,
        predicted_values,
        reference_name="true_values",
        estimate_name="predicted_values",
    )

    squared_error = (predicted_array - true_array) ** 2

    return float(np.sqrt(np.mean(squared_error)))


def mae(
    true_values: ArrayLike,
    predicted_values: ArrayLike,
) -> float:
    """Return the mean absolute error."""
    true_array, predicted_array = _validate_matching_arrays(
        true_values,
        predicted_values,
        reference_name="true_values",
        estimate_name="predicted_values",
    )

    return float(
        np.mean(np.abs(predicted_array - true_array))
    )


def relative_l2_error(
    true_values: ArrayLike,
    predicted_values: ArrayLike,
) -> float:
    """Return the relative Euclidean error.

    The metric is

    ``||predicted - true||₂ / ||true||₂``.

    When the true array has zero norm, the function returns zero if
    both arrays are zero and infinity otherwise.
    """
    true_array, predicted_array = _validate_matching_arrays(
        true_values,
        predicted_values,
        reference_name="true_values",
        estimate_name="predicted_values",
    )

    error_norm = np.linalg.norm(
        predicted_array.ravel(order="C")
        - true_array.ravel(order="C")
    )
    true_norm = np.linalg.norm(
        true_array.ravel(order="C")
    )

    if true_norm == 0.0:
        if error_norm == 0.0:
            return 0.0

        return float("inf")

    return float(error_norm / true_norm)


def max_absolute_error(
    true_values: ArrayLike,
    predicted_values: ArrayLike,
) -> float:
    """Return the largest absolute pointwise error."""
    true_array, predicted_array = _validate_matching_arrays(
        true_values,
        predicted_values,
        reference_name="true_values",
        estimate_name="predicted_values",
    )

    return float(
        np.max(np.abs(predicted_array - true_array))
    )


def residual_norm(
    predicted_measurements: ArrayLike,
    observed_measurements: ArrayLike,
) -> float:
    """Return the Euclidean measurement residual norm."""
    predicted_array, observed_array = _validate_measurement_arrays(
        predicted_measurements,
        observed_measurements,
    )

    return float(np.linalg.norm(predicted_array - observed_array))


def relative_residual(
    predicted_measurements: ArrayLike,
    observed_measurements: ArrayLike,
) -> float:
    """Return the residual norm relative to the observed-data norm.

    When the observed norm is zero, the function returns zero for a
    zero residual and infinity for a nonzero residual.
    """
    predicted_array, observed_array = _validate_measurement_arrays(
        predicted_measurements,
        observed_measurements,
    )

    residual_value = float(
        np.linalg.norm(predicted_array - observed_array)
    )
    observed_norm = float(np.linalg.norm(observed_array))

    if observed_norm == 0.0:
        if residual_value == 0.0:
            return 0.0

        return float("inf")

    return residual_value / observed_norm


def residual_rms(
    predicted_measurements: ArrayLike,
    observed_measurements: ArrayLike,
) -> float:
    """Return the root-mean-square measurement residual."""
    predicted_array, observed_array = _validate_measurement_arrays(
        predicted_measurements,
        observed_measurements,
    )

    residual_value = float(
        np.linalg.norm(predicted_array - observed_array)
    )

    return residual_value / float(np.sqrt(predicted_array.size))


def compute_error_field(
    true_source: ArrayLike,
    reconstructed_source: ArrayLike,
) -> NDArray[np.float64]:
    """Return the signed interior reconstruction error field.

    Interior entries contain ``reconstructed_source - true_source``.
    Boundary entries are zero because boundary source values are not
    unknowns in the inverse problem.
    """
    true_array, reconstructed_array = _validate_source_fields(
        true_source,
        reconstructed_source,
    )

    error_field = np.zeros_like(reconstructed_array, dtype=float)
    error_field[1:-1, 1:-1] = (
        reconstructed_array[1:-1, 1:-1]
        - true_array[1:-1, 1:-1]
    )

    return error_field


def compute_all_metrics(
    true_source: ArrayLike,
    reconstructed_source: ArrayLike,
) -> dict[str, float]:
    """Compute source-reconstruction metrics on interior nodes."""
    true_interior, reconstructed_interior = _source_interiors(
        true_source,
        reconstructed_source,
    )

    return {
        "rmse": rmse(true_interior, reconstructed_interior),
        "mae": mae(true_interior, reconstructed_interior),
        "relative_l2_error": relative_l2_error(
            true_interior,
            reconstructed_interior,
        ),
        "max_absolute_error": max_absolute_error(
            true_interior,
            reconstructed_interior,
        ),
    }


def validate_reconstruction(
    true_source: ArrayLike,
    reconstructed_source: ArrayLike,
) -> dict[str, float]:
    """Validate a reconstruction against known ground truth.

    This function is mainly intended for synthetic benchmark mode,
    where the true heat-source field is available.
    """
    return compute_all_metrics(
        true_source,
        reconstructed_source,
    )