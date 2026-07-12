"""Sparse sensor handling for ThermoReconLab.

This module contains the sensor data structure, sensor placement
strategies, field sampling, and controlled measurement noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from thermoreconlab.core.fields import ensure_2d_array, validate_field
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class SensorData:
    """Store sensor indices and their measured values.

    Parameters
    ----------
    indices:
        Integer grid indices with shape ``(n_sensors, 2)``.
        Each row contains ``(i, j)``.
    values:
        Sensor measurements with shape ``(n_sensors,)``.
    coordinates:
        Optional physical coordinates with shape ``(n_sensors, 2)``.
    """

    indices: NDArray[np.int64]
    values: NDArray[np.float64]
    coordinates: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate and copy all sensor data."""
        indices = _validate_index_array(self.indices)
        values = _validate_measurements(self.values)

        if len(indices) != len(values):
            raise ValidationError(
                "indices and values must contain the same "
                "number of sensors."
            )

        coordinates: NDArray[np.float64] | None = None

        if self.coordinates is not None:
            coordinates = np.asarray(self.coordinates, dtype=float)

            if coordinates.shape != indices.shape:
                raise ValidationError(
                    "coordinates must have shape "
                    f"{indices.shape}, but received "
                    f"{coordinates.shape}."
                )

            if not np.all(np.isfinite(coordinates)):
                raise ValidationError(
                    "coordinates must contain only finite values."
                )

            coordinates = coordinates.copy()

        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coordinates", coordinates)

    def __len__(self) -> int:
        """Return the number of sensors."""
        return len(self.values)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert sensor measurements to a pandas DataFrame."""
        data: dict[str, Any] = {
            "i": self.indices[:, 0],
            "j": self.indices[:, 1],
            "value": self.values,
        }

        if self.coordinates is not None:
            data["x"] = self.coordinates[:, 0]
            data["y"] = self.coordinates[:, 1]

        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> SensorData:
        """Create sensor data from a pandas DataFrame.

        Required columns are ``i``, ``j``, and ``value``.
        Optional coordinate columns are ``x`` and ``y``.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValidationError(
                "dataframe must be a pandas DataFrame."
            )

        required_columns = {"i", "j", "value"}
        missing_columns = required_columns.difference(dataframe.columns)

        if missing_columns:
            raise ValidationError(
                "Sensor DataFrame is missing required columns: "
                f"{sorted(missing_columns)}."
            )

        indices = dataframe[["i", "j"]].to_numpy()
        values = dataframe["value"].to_numpy()

        coordinates = None
        coordinate_columns = {"x", "y"}

        if coordinate_columns.issubset(dataframe.columns):
            coordinates = dataframe[["x", "y"]].to_numpy()

        return cls(
            indices=indices,
            values=values,
            coordinates=coordinates,
        )


def _validate_positive_count(
    count: int,
    maximum: int,
) -> int:
    """Validate a requested sensor count."""
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise ValidationError("count must be an integer.")

    count = int(count)

    if count <= 0:
        raise ValidationError("count must be greater than zero.")

    if count > maximum:
        raise ValidationError(
            f"count cannot exceed the {maximum} available grid points."
        )

    return count


def _validate_nonnegative_real(
    value: Real,
    name: str,
) -> float:
    """Validate a finite nonnegative real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number.")

    result = float(value)

    if not np.isfinite(result):
        raise ValidationError(f"{name} must be finite.")

    if result < 0.0:
        raise ValidationError(f"{name} must be nonnegative.")

    return result


def _validate_index_array(
    indices: ArrayLike,
    *,
    grid: Grid2D | None = None,
    reject_duplicates: bool = True,
) -> NDArray[np.int64]:
    """Validate an array of ``(i, j)`` grid indices."""
    try:
        array = np.asarray(indices)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "sensor indices could not be converted to an array."
        ) from error

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValidationError(
            "sensor indices must have shape (n_sensors, 2)."
        )

    if len(array) == 0:
        raise ValidationError(
            "sensor indices must contain at least one sensor."
        )

    try:
        numeric_array = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "sensor indices must contain numeric values."
        ) from error

    if not np.all(np.isfinite(numeric_array)):
        raise ValidationError(
            "sensor indices must contain only finite values."
        )

    if not np.all(numeric_array == np.floor(numeric_array)):
        raise ValidationError(
            "sensor indices must contain integer values."
        )

    clean_indices = numeric_array.astype(np.int64)

    if reject_duplicates:
        unique_indices = np.unique(clean_indices, axis=0)

        if len(unique_indices) != len(clean_indices):
            raise ValidationError(
                "sensor indices must not contain duplicates."
            )

    if grid is not None:
        if not isinstance(grid, Grid2D):
            raise ValidationError("grid must be a Grid2D object.")

        i_values = clean_indices[:, 0]
        j_values = clean_indices[:, 1]

        if np.any(i_values < 0) or np.any(i_values >= grid.nx):
            raise ValidationError(
                f"sensor i-indices must satisfy 0 <= i < {grid.nx}."
            )

        if np.any(j_values < 0) or np.any(j_values >= grid.ny):
            raise ValidationError(
                f"sensor j-indices must satisfy 0 <= j < {grid.ny}."
            )

    return clean_indices.copy()


def _validate_measurements(
    values: ArrayLike,
) -> NDArray[np.float64]:
    """Validate one-dimensional sensor measurements."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "sensor values must contain numeric values."
        ) from error

    if array.ndim != 1:
        raise ValidationError(
            "sensor values must be one-dimensional."
        )

    if array.size == 0:
        raise ValidationError(
            "sensor values must not be empty."
        )

    if not np.all(np.isfinite(array)):
        raise ValidationError(
            "sensor values must contain only finite values."
        )

    return array.copy()


def _candidate_indices(
    grid: Grid2D,
    *,
    include_boundary: bool,
) -> NDArray[np.int64]:
    """Return all available candidate sensor indices."""
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    if include_boundary:
        i_values = np.arange(grid.nx)
        j_values = np.arange(grid.ny)
    else:
        i_values = np.arange(1, grid.nx - 1)
        j_values = np.arange(1, grid.ny - 1)

    i_mesh, j_mesh = np.meshgrid(
        i_values,
        j_values,
        indexing="ij",
    )

    return np.column_stack(
        (
            i_mesh.ravel(order="C"),
            j_mesh.ravel(order="C"),
        )
    ).astype(np.int64)


def custom_sensors(
    indices: ArrayLike,
    grid: Grid2D,
) -> NDArray[np.int64]:
    """Validate and return user-provided sensor indices."""
    return _validate_index_array(indices, grid=grid)


def regular_grid_sensors(
    grid: Grid2D,
    count: int,
    *,
    include_boundary: bool = False,
) -> NDArray[np.int64]:
    """Place sensors in a near-regular rectangular arrangement."""
    candidates = _candidate_indices(
        grid,
        include_boundary=include_boundary,
    )

    count = _validate_positive_count(count, len(candidates))

    if include_boundary:
        i_min, i_max = 0, grid.nx - 1
        j_min, j_max = 0, grid.ny - 1
    else:
        i_min, i_max = 1, grid.nx - 2
        j_min, j_max = 1, grid.ny - 2

    available_i = i_max - i_min + 1
    available_j = j_max - j_min + 1

    aspect_ratio = available_i / available_j

    number_i = min(
        available_i,
        max(1, int(np.ceil(np.sqrt(count * aspect_ratio)))),
    )

    number_j = min(
        available_j,
        max(1, int(np.ceil(count / number_i))),
    )

    while number_i * number_j < count:
        if number_j < available_j:
            number_j += 1
        elif number_i < available_i:
            number_i += 1
        else:
            break

    selected_i = np.rint(
        np.linspace(i_min, i_max, number_i)
    ).astype(int)

    selected_j = np.rint(
        np.linspace(j_min, j_max, number_j)
    ).astype(int)

    i_mesh, j_mesh = np.meshgrid(
        selected_i,
        selected_j,
        indexing="ij",
    )

    regular_indices = np.column_stack(
        (
            i_mesh.ravel(order="C"),
            j_mesh.ravel(order="C"),
        )
    )

    selection = np.linspace(
        0,
        len(regular_indices) - 1,
        count,
        dtype=int,
    )

    return _validate_index_array(
        regular_indices[selection],
        grid=grid,
    )


def random_sensors(
    grid: Grid2D,
    count: int,
    *,
    seed: int | None = None,
    include_boundary: bool = False,
) -> NDArray[np.int64]:
    """Select unique sensor locations randomly."""
    candidates = _candidate_indices(
        grid,
        include_boundary=include_boundary,
    )

    count = _validate_positive_count(count, len(candidates))

    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "seed must be an integer or None."
        ) from error

    selected = rng.choice(
        len(candidates),
        size=count,
        replace=False,
    )

    return candidates[selected].copy()


def boundary_sensors(
    grid: Grid2D,
    count: int,
) -> NDArray[np.int64]:
    """Place sensors evenly around the outer grid boundary."""
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    boundary: list[tuple[int, int]] = []

    boundary.extend((0, j) for j in range(grid.ny))
    boundary.extend(
        (i, grid.ny - 1)
        for i in range(1, grid.nx)
    )
    boundary.extend(
        (grid.nx - 1, j)
        for j in range(grid.ny - 2, -1, -1)
    )
    boundary.extend(
        (i, 0)
        for i in range(grid.nx - 2, 0, -1)
    )

    boundary_array = np.asarray(boundary, dtype=np.int64)
    count = _validate_positive_count(count, len(boundary_array))

    selection = np.linspace(
        0,
        len(boundary_array) - 1,
        count,
        dtype=int,
    )

    return boundary_array[selection].copy()


def center_focused_sensors(
    grid: Grid2D,
    count: int,
    *,
    seed: int | None = None,
    include_boundary: bool = False,
) -> NDArray[np.int64]:
    """Randomly place sensors with higher probability near the center."""
    candidates = _candidate_indices(
        grid,
        include_boundary=include_boundary,
    )

    count = _validate_positive_count(count, len(candidates))

    x_coordinates = grid.x[candidates[:, 0]]
    y_coordinates = grid.y[candidates[:, 1]]

    normalized_x = (
        x_coordinates / grid.domain.length_x - 0.5
    )
    normalized_y = (
        y_coordinates / grid.domain.length_y - 0.5
    )

    squared_distance = normalized_x**2 + normalized_y**2
    weights = np.exp(-8.0 * squared_distance)
    weights = weights / np.sum(weights)

    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "seed must be an integer or None."
        ) from error

    selected = rng.choice(
        len(candidates),
        size=count,
        replace=False,
        p=weights,
    )

    return candidates[selected].copy()


def sample_field(
    field: ArrayLike,
    sensor_indices: ArrayLike,
    grid: Grid2D | None = None,
) -> NDArray[np.float64]:
    """Sample field values at specified grid indices."""
    if grid is not None:
        field_array = validate_field(field, grid)
        indices = _validate_index_array(
            sensor_indices,
            grid=grid,
        )
    else:
        field_array = ensure_2d_array(field)
        indices = _validate_index_array(sensor_indices)

        if (
            np.any(indices[:, 0] < 0)
            or np.any(indices[:, 0] >= field_array.shape[0])
            or np.any(indices[:, 1] < 0)
            or np.any(indices[:, 1] >= field_array.shape[1])
        ):
            raise ValidationError(
                "sensor indices lie outside the field."
            )

    return field_array[
        indices[:, 0],
        indices[:, 1],
    ].astype(float, copy=True)


def create_sensor_data(
    field: ArrayLike,
    sensor_indices: ArrayLike,
    grid: Grid2D | None = None,
) -> SensorData:
    """Sample a field and return a SensorData object."""
    values = sample_field(
        field,
        sensor_indices,
        grid=grid,
    )

    indices = _validate_index_array(
        sensor_indices,
        grid=grid,
    )

    coordinates = None

    if grid is not None:
        coordinates = np.column_stack(
            (
                grid.x[indices[:, 0]],
                grid.y[indices[:, 1]],
            )
        )

    return SensorData(
        indices=indices,
        values=values,
        coordinates=coordinates,
    )


def add_gaussian_noise(
    values: ArrayLike,
    noise_level: Real = 0.01,
    *,
    seed: int | None = None,
    relative: bool = True,
) -> NDArray[np.float64]:
    """Add reproducible Gaussian noise to sensor measurements.

    For relative noise, the standard deviation is

    ``noise_level * max(abs(values))``.

    For absolute noise, ``noise_level`` is used directly as the
    standard deviation.
    """
    clean_values = _validate_measurements(values)
    noise_level = _validate_nonnegative_real(
        noise_level,
        "noise_level",
    )

    if relative:
        scale = float(np.max(np.abs(clean_values)))
        standard_deviation = noise_level * scale
    else:
        standard_deviation = noise_level

    if standard_deviation == 0.0:
        return clean_values.copy()

    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "seed must be an integer or None."
        ) from error

    noise = rng.normal(
        loc=0.0,
        scale=standard_deviation,
        size=clean_values.shape,
    )

    return clean_values + noise


def add_absolute_noise(
    values: ArrayLike,
    sigma: Real = 0.01,
    *,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Add Gaussian noise with an absolute standard deviation."""
    return add_gaussian_noise(
        values,
        noise_level=sigma,
        seed=seed,
        relative=False,
    )


def add_noise_to_sensor_data(
    sensor_data: SensorData,
    noise_level: Real = 0.01,
    *,
    seed: int | None = None,
    relative: bool = True,
) -> SensorData:
    """Return a new SensorData object with noisy measurements."""
    if not isinstance(sensor_data, SensorData):
        raise ValidationError(
            "sensor_data must be a SensorData object."
        )

    noisy_values = add_gaussian_noise(
        sensor_data.values,
        noise_level=noise_level,
        seed=seed,
        relative=relative,
    )

    return SensorData(
        indices=sensor_data.indices,
        values=noisy_values,
        coordinates=sensor_data.coordinates,
    )