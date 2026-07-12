"""Tests for field validation and conversion helpers."""

import numpy as np
import pytest

from thermoreconlab.core.fields import (
    ensure_2d_array,
    flatten_field,
    reshape_field,
    validate_field,
)
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.exceptions import ValidationError


def test_ensure_2d_array_converts_values_to_float() -> None:
    """Numeric nested lists should become floating-point arrays."""
    field = [[1, 2], [3, 4]]

    result = ensure_2d_array(field)

    assert result.shape == (2, 2)
    assert result.dtype == np.float64
    assert np.array_equal(
        result,
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_ensure_2d_array_returns_copy() -> None:
    """The returned field should not share memory with its input."""
    field = np.ones((3, 3))

    result = ensure_2d_array(field)
    result[0, 0] = 5.0

    assert field[0, 0] == 1.0


@pytest.mark.parametrize(
    "invalid_field",
    [
        np.array([1.0, 2.0, 3.0]),
        np.zeros((2, 2, 2)),
        3.0,
    ],
)
def test_ensure_2d_array_rejects_invalid_dimensions(
    invalid_field: object,
) -> None:
    """Only two-dimensional field inputs should be accepted."""
    with pytest.raises(ValidationError):
        ensure_2d_array(invalid_field)


def test_ensure_2d_array_rejects_empty_field() -> None:
    """Empty two-dimensional arrays should be rejected."""
    empty_field = np.empty((0, 0))

    with pytest.raises(ValidationError):
        ensure_2d_array(empty_field)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_ensure_2d_array_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    """Fields must not contain NaN or infinite values."""
    field = np.zeros((3, 3))
    field[1, 1] = invalid_value

    with pytest.raises(ValidationError):
        ensure_2d_array(field)


def test_ensure_2d_array_rejects_non_numeric_values() -> None:
    """Text values that cannot be converted should be rejected."""
    field = [["one", "two"], ["three", "four"]]

    with pytest.raises(ValidationError):
        ensure_2d_array(field)


def test_validate_field_accepts_matching_shape() -> None:
    """A field matching the grid shape should be accepted."""
    grid = Grid2D(nx=4, ny=5)
    field = np.ones(grid.shape)

    result = validate_field(field, grid)

    assert result.shape == grid.shape
    assert np.array_equal(result, field)


def test_validate_field_rejects_wrong_shape() -> None:
    """A field with the wrong shape should be rejected."""
    grid = Grid2D(nx=4, ny=5)
    field = np.ones((5, 4))

    with pytest.raises(ValidationError):
        validate_field(field, grid)


def test_validate_field_rejects_invalid_grid() -> None:
    """The grid argument must be a Grid2D object."""
    field = np.ones((3, 3))

    with pytest.raises(ValidationError):
        validate_field(
            field,
            "invalid grid",  # type: ignore[arg-type]
        )


def test_flatten_field_uses_c_order() -> None:
    """Flattening should preserve the agreed C-order convention."""
    field = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    result = flatten_field(field)

    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert result.shape == (6,)
    assert np.array_equal(result, expected)


def test_reshape_field_restores_grid_shape() -> None:
    """A valid vector should be reshaped to the grid field shape."""
    grid = Grid2D(nx=3, ny=4)
    vector = np.arange(grid.size, dtype=float)

    result = reshape_field(vector, grid)

    assert result.shape == grid.shape
    assert np.array_equal(
        result.flatten(order="C"),
        vector,
    )


def test_flatten_and_reshape_round_trip() -> None:
    """Flattening followed by reshaping should preserve values."""
    grid = Grid2D(nx=4, ny=6)
    field = np.arange(grid.size, dtype=float).reshape(grid.shape)

    vector = flatten_field(field)
    restored = reshape_field(vector, grid)

    assert np.array_equal(restored, field)


def test_reshape_field_rejects_non_vector_input() -> None:
    """Only one-dimensional inputs should be reshaped."""
    grid = Grid2D(nx=3, ny=3)
    matrix = np.zeros(grid.shape)

    with pytest.raises(ValidationError):
        reshape_field(matrix, grid)


def test_reshape_field_rejects_wrong_size() -> None:
    """The vector must contain one value per grid point."""
    grid = Grid2D(nx=3, ny=4)
    vector = np.zeros(grid.size - 1)

    with pytest.raises(ValidationError):
        reshape_field(vector, grid)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_reshape_field_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    """Field vectors must contain only finite values."""
    grid = Grid2D(nx=3, ny=3)
    vector = np.zeros(grid.size)
    vector[4] = invalid_value

    with pytest.raises(ValidationError):
        reshape_field(vector, grid)