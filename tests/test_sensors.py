"""Tests for sensor placement, sampling, and noise."""

import numpy as np
import pandas as pd
import pytest

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.sensors import (
    SensorData,
    add_absolute_noise,
    add_gaussian_noise,
    add_noise_to_sensor_data,
    boundary_sensors,
    center_focused_sensors,
    create_sensor_data,
    custom_sensors,
    random_sensors,
    regular_grid_sensors,
    sample_field,
)


def test_sensor_data_stores_valid_measurements() -> None:
    """Valid sensor arrays should be stored correctly."""
    indices = np.array([[1, 2], [3, 4]])
    values = np.array([0.5, 1.5])

    sensor_data = SensorData(indices=indices, values=values)

    assert len(sensor_data) == 2
    assert sensor_data.indices.dtype == np.int64
    assert sensor_data.values.dtype == np.float64


def test_sensor_data_rejects_mismatched_lengths() -> None:
    """Indices and measurements must have matching lengths."""
    with pytest.raises(ValidationError):
        SensorData(
            indices=np.array([[1, 2], [3, 4]]),
            values=np.array([1.0]),
        )


def test_sensor_data_dataframe_round_trip() -> None:
    """SensorData should convert to and from DataFrames."""
    original = SensorData(
        indices=np.array([[1, 2], [3, 4]]),
        values=np.array([0.5, 1.5]),
        coordinates=np.array([[0.1, 0.2], [0.3, 0.4]]),
    )

    dataframe = original.to_dataframe()
    restored = SensorData.from_dataframe(dataframe)

    assert isinstance(dataframe, pd.DataFrame)
    assert np.array_equal(restored.indices, original.indices)
    assert np.allclose(restored.values, original.values)
    assert np.allclose(
        restored.coordinates,
        original.coordinates,
    )


def test_custom_sensors_accept_valid_indices() -> None:
    """Valid custom indices should be returned."""
    grid = Grid2D(nx=6, ny=7)
    indices = np.array([[1, 1], [2, 3], [4, 5]])

    result = custom_sensors(indices, grid)

    assert np.array_equal(result, indices)


def test_custom_sensors_reject_duplicates() -> None:
    """Duplicate sensor locations should be rejected."""
    grid = Grid2D(nx=6, ny=7)

    with pytest.raises(ValidationError):
        custom_sensors(
            np.array([[1, 1], [1, 1]]),
            grid,
        )


def test_custom_sensors_reject_out_of_range_indices() -> None:
    """Sensor locations must lie inside the grid."""
    grid = Grid2D(nx=6, ny=7)

    with pytest.raises(ValidationError):
        custom_sensors(
            np.array([[1, 1], [6, 3]]),
            grid,
        )


def test_regular_grid_sensors_returns_requested_count() -> None:
    """Regular placement should return exactly the requested count."""
    grid = Grid2D(nx=15, ny=17)

    indices = regular_grid_sensors(grid, count=12)

    assert indices.shape == (12, 2)
    assert len(np.unique(indices, axis=0)) == 12


def test_regular_sensors_exclude_boundary_by_default() -> None:
    """Default regular sensors should use interior nodes."""
    grid = Grid2D(nx=10, ny=12)

    indices = regular_grid_sensors(grid, count=9)

    assert np.all(indices[:, 0] > 0)
    assert np.all(indices[:, 0] < grid.nx - 1)
    assert np.all(indices[:, 1] > 0)
    assert np.all(indices[:, 1] < grid.ny - 1)


def test_random_sensors_are_reproducible() -> None:
    """The same seed should return the same locations."""
    grid = Grid2D(nx=12, ny=12)

    first = random_sensors(grid, count=15, seed=42)
    second = random_sensors(grid, count=15, seed=42)

    assert np.array_equal(first, second)


def test_random_sensors_have_no_duplicates() -> None:
    """Random placement must sample without replacement."""
    grid = Grid2D(nx=12, ny=12)

    indices = random_sensors(grid, count=30, seed=4)

    assert len(np.unique(indices, axis=0)) == 30


def test_boundary_sensors_lie_on_boundary() -> None:
    """All boundary sensors should be on at least one edge."""
    grid = Grid2D(nx=9, ny=11)

    indices = boundary_sensors(grid, count=15)

    on_boundary = (
        (indices[:, 0] == 0)
        | (indices[:, 0] == grid.nx - 1)
        | (indices[:, 1] == 0)
        | (indices[:, 1] == grid.ny - 1)
    )

    assert np.all(on_boundary)
    assert len(np.unique(indices, axis=0)) == 15


def test_center_focused_sensors_return_requested_count() -> None:
    """Center-focused placement should return unique indices."""
    grid = Grid2D(nx=15, ny=15)

    indices = center_focused_sensors(
        grid,
        count=20,
        seed=42,
    )

    assert indices.shape == (20, 2)
    assert len(np.unique(indices, axis=0)) == 20


def test_sample_field_returns_correct_values() -> None:
    """Sampling must return field[i, j] values."""
    grid = Grid2D(nx=4, ny=5)
    field = np.arange(grid.size).reshape(grid.shape)
    indices = np.array([[0, 0], [1, 3], [3, 4]])

    values = sample_field(field, indices, grid)

    expected = np.array([
        field[0, 0],
        field[1, 3],
        field[3, 4],
    ])

    assert np.array_equal(values, expected)


def test_sample_field_rejects_invalid_indices() -> None:
    """Sampling should reject locations outside the field."""
    field = np.ones((4, 5))

    with pytest.raises(ValidationError):
        sample_field(
            field,
            np.array([[1, 1], [4, 2]]),
        )


def test_create_sensor_data_includes_coordinates() -> None:
    """Physical coordinates should be included when grid is given."""
    grid = Grid2D(nx=5, ny=6)
    field = np.ones(grid.shape)
    indices = np.array([[1, 2], [3, 4]])

    sensor_data = create_sensor_data(
        field,
        indices,
        grid,
    )

    assert isinstance(sensor_data, SensorData)
    assert sensor_data.coordinates is not None

    assert sensor_data.coordinates[0, 0] == pytest.approx(
        grid.x[1]
    )
    assert sensor_data.coordinates[0, 1] == pytest.approx(
        grid.y[2]
    )


def test_gaussian_noise_preserves_shape() -> None:
    """Noise addition should preserve measurement shape."""
    values = np.ones(10)

    noisy = add_gaussian_noise(
        values,
        noise_level=0.02,
        seed=42,
    )

    assert noisy.shape == values.shape


def test_gaussian_noise_is_reproducible() -> None:
    """The same seed should produce the same noise."""
    values = np.linspace(0.0, 1.0, 20)

    first = add_gaussian_noise(values, seed=42)
    second = add_gaussian_noise(values, seed=42)

    assert np.array_equal(first, second)


def test_zero_noise_returns_unchanged_copy() -> None:
    """A zero noise level should preserve measurements."""
    values = np.array([1.0, 2.0, 3.0])

    result = add_gaussian_noise(
        values,
        noise_level=0.0,
        seed=42,
    )

    assert np.array_equal(result, values)
    assert result is not values


def test_noise_does_not_modify_input() -> None:
    """Noise addition must not change the original array."""
    values = np.ones(10)
    original = values.copy()

    add_absolute_noise(values, sigma=0.1, seed=42)

    assert np.array_equal(values, original)


def test_negative_noise_level_raises_error() -> None:
    """Noise levels must be nonnegative."""
    with pytest.raises(ValidationError):
        add_gaussian_noise(
            np.ones(5),
            noise_level=-0.1,
        )


def test_add_noise_to_sensor_data_returns_new_object() -> None:
    """Noisy sensor data should preserve locations."""
    sensor_data = SensorData(
        indices=np.array([[1, 1], [2, 2]]),
        values=np.array([1.0, 2.0]),
    )

    noisy = add_noise_to_sensor_data(
        sensor_data,
        noise_level=0.1,
        seed=42,
    )

    assert isinstance(noisy, SensorData)
    assert noisy is not sensor_data
    assert np.array_equal(noisy.indices, sensor_data.indices)
    assert not np.array_equal(noisy.values, sensor_data.values)


@pytest.mark.parametrize("invalid_count", [0, -1, 1.5, True])
def test_sensor_placement_rejects_invalid_count(
    invalid_count: object,
) -> None:
    """Sensor count must be a positive integer."""
    grid = Grid2D(nx=10, ny=10)

    with pytest.raises(ValidationError):
        random_sensors(
            grid,
            count=invalid_count,  # type: ignore[arg-type]
        )