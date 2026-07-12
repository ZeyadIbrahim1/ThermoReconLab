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
    predicted_array, observed_array = _validate_matching_arrays(
        predicted_measurements,
        observed_measurements,
        reference_name="predicted_measurements",
        estimate_name="observed_measurements",
    )

    residual = (
        predicted_array.ravel(order="C")
        - observed_array.ravel(order="C")
    )

    return float(np.linalg.norm(residual))


def compute_error_field(
    true_source: ArrayLike,
    reconstructed_source: ArrayLike,
) -> NDArray[np.float64]:
    """Return the signed reconstruction error field.

    The error is defined as

    ``reconstructed_source - true_source``.
    """
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

    return reconstructed_array - true_array


def compute_all_metrics(
    true_source: ArrayLike,
    reconstructed_source: ArrayLike,
) -> dict[str, float]:
    """Compute the standard source-reconstruction metrics."""
    return {
        "rmse": rmse(true_source, reconstructed_source),
        "mae": mae(true_source, reconstructed_source),
        "relative_l2_error": relative_l2_error(
            true_source,
            reconstructed_source,
        ),
        "max_absolute_error": max_absolute_error(
            true_source,
            reconstructed_source,
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