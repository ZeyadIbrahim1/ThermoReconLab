"""Field validation and conversion helpers for ThermoReconLab."""


import numpy as np
from numpy.typing import ArrayLike, NDArray

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.exceptions import ValidationError


def ensure_2d_array(
    field: ArrayLike,
    *,
    name: str = "field",
) -> NDArray[np.float64]:
    """Convert input data to a finite two-dimensional float array.

    Parameters
    ----------
    field:
        Array-like input representing a two-dimensional field.
    name:
        Name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        A two-dimensional NumPy array with floating-point values.

    Raises
    ------
    ValidationError
        If the input cannot be converted to a numeric array, is not
        two-dimensional, is empty, or contains non-finite values.
    """
    try:
        array = np.asarray(field, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"{name} must contain numeric values."
        ) from error

    if array.ndim != 2:
        raise ValidationError(
            f"{name} must be two-dimensional, "
            f"but received an array with {array.ndim} dimensions."
        )

    if array.size == 0:
        raise ValidationError(f"{name} must not be empty.")

    if not np.all(np.isfinite(array)):
        raise ValidationError(
            f"{name} must contain only finite values."
        )

    return array.copy()


def validate_field(
    field: ArrayLike,
    grid: Grid2D,
    *,
    name: str = "field",
) -> NDArray[np.float64]:
    """Validate that a field is compatible with a grid.

    Parameters
    ----------
    field:
        Array-like input representing a two-dimensional field.
    grid:
        Grid that defines the required field shape.
    name:
        Name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        Validated floating-point field with shape ``grid.shape``.

    Raises
    ------
    ValidationError
        If the grid is invalid or the field does not match its shape.
    """
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    array = ensure_2d_array(field, name=name)

    if array.shape != grid.shape:
        raise ValidationError(
            f"{name} must have shape {grid.shape}, "
            f"but received {array.shape}."
        )

    return array


def flatten_field(
    field: ArrayLike,
    *,
    name: str = "field",
) -> NDArray[np.float64]:
    """Flatten a valid two-dimensional field using C-order indexing.

    Parameters
    ----------
    field:
        Two-dimensional field to flatten.
    name:
        Name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        One-dimensional copy of the field.

    Notes
    -----
    C-order flattening is used consistently throughout the package so
    that field vectors match the indexing convention used by Grid2D.
    """
    array = ensure_2d_array(field, name=name)
    return array.reshape(array.size, order="C").copy()


def reshape_field(
    vector: ArrayLike,
    grid: Grid2D,
    *,
    name: str = "vector",
) -> NDArray[np.float64]:
    """Reshape a field vector into the shape of a grid.

    Parameters
    ----------
    vector:
        One-dimensional array containing one value per grid point.
    grid:
        Grid defining the target field shape.
    name:
        Name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        Two-dimensional field with shape ``grid.shape``.

    Raises
    ------
    ValidationError
        If the grid is invalid, the input is not one-dimensional,
        its size is incorrect, or it contains non-finite values.
    """
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    try:
        array = np.asarray(vector, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"{name} must contain numeric values."
        ) from error

    if array.ndim != 1:
        raise ValidationError(
            f"{name} must be one-dimensional."
        )

    if array.size != grid.size:
        raise ValidationError(
            f"{name} must contain {grid.size} values, "
            f"but received {array.size}."
        )

    if not np.all(np.isfinite(array)):
        raise ValidationError(
            f"{name} must contain only finite values."
        )

    return array.reshape(grid.shape, order="C").copy()