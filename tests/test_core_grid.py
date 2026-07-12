"""Tests for the structured two-dimensional grid."""

import numpy as np
import pytest

from thermoreconlab.core.domain import Domain2D
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.exceptions import ValidationError


def test_default_grid_properties() -> None:
    """The default grid should cover the unit square."""
    grid = Grid2D()

    assert grid.shape == (50, 50)
    assert grid.size == 2500
    assert grid.dx == pytest.approx(1.0 / 49.0)
    assert grid.dy == pytest.approx(1.0 / 49.0)

    assert grid.x.shape == (50,)
    assert grid.y.shape == (50,)
    assert grid.X.shape == (50, 50)
    assert grid.Y.shape == (50, 50)


def test_rectangular_grid_spacing() -> None:
    """Grid spacing should use the physical domain lengths."""
    domain = Domain2D(length_x=2.0, length_y=3.0)
    grid = Grid2D(nx=5, ny=7, domain=domain)

    assert grid.shape == (5, 7)
    assert grid.size == 35
    assert grid.dx == pytest.approx(0.5)
    assert grid.dy == pytest.approx(0.5)

    assert grid.x[0] == pytest.approx(0.0)
    assert grid.x[-1] == pytest.approx(2.0)
    assert grid.y[0] == pytest.approx(0.0)
    assert grid.y[-1] == pytest.approx(3.0)


def test_mesh_uses_ij_indexing() -> None:
    """Mesh arrays should follow the package field convention."""
    grid = Grid2D(nx=4, ny=5)

    X, Y = grid.mesh()

    assert X.shape == (4, 5)
    assert Y.shape == (4, 5)

    assert np.allclose(X[:, 0], grid.x)
    assert np.allclose(Y[0, :], grid.y)


def test_minimum_grid_size_is_allowed() -> None:
    """A 3 by 3 grid should be valid."""
    grid = Grid2D(nx=3, ny=3)

    assert grid.shape == (3, 3)
    assert grid.size == 9


@pytest.mark.parametrize("invalid_value", [0, 1, 2, -1])
def test_too_few_grid_points_raise_error(
    invalid_value: int,
) -> None:
    """Each grid direction must contain at least three points."""
    with pytest.raises(ValidationError):
        Grid2D(nx=invalid_value)


@pytest.mark.parametrize("invalid_value", [3.5, "5", None, True])
def test_non_integer_grid_points_raise_error(
    invalid_value: object,
) -> None:
    """Grid point counts must be integers."""
    with pytest.raises(ValidationError):
        Grid2D(nx=invalid_value)  # type: ignore[arg-type]


def test_invalid_domain_raises_error() -> None:
    """The domain must be represented by Domain2D."""
    with pytest.raises(ValidationError):
        Grid2D(domain="unit square")  # type: ignore[arg-type]


def test_flatten_and_reshape_round_trip() -> None:
    """Flattening and reshaping should preserve field values."""
    grid = Grid2D(nx=4, ny=5)
    field_array = np.arange(grid.size, dtype=float).reshape(grid.shape)

    vector = grid.flatten(field_array)
    restored = grid.reshape(vector)

    assert vector.shape == (grid.size,)
    assert restored.shape == grid.shape
    assert np.array_equal(restored, field_array)


def test_flatten_rejects_wrong_shape() -> None:
    """Fields must match the grid shape before flattening."""
    grid = Grid2D(nx=4, ny=5)
    wrong_shape = np.zeros((5, 4))

    with pytest.raises(ValidationError):
        grid.flatten(wrong_shape)


def test_reshape_rejects_wrong_vector_length() -> None:
    """Vectors must contain exactly one value per grid point."""
    grid = Grid2D(nx=4, ny=5)
    wrong_length = np.zeros(grid.size - 1)

    with pytest.raises(ValidationError):
        grid.reshape(wrong_length)


def test_reshape_rejects_non_vector_input() -> None:
    """The reshape helper should require one-dimensional input."""
    grid = Grid2D(nx=4, ny=5)
    matrix = np.zeros((4, 5))

    with pytest.raises(ValidationError):
        grid.reshape(matrix)


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_flatten_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    """Fields containing NaN or infinity should be rejected."""
    grid = Grid2D(nx=3, ny=3)
    field_array = np.zeros(grid.shape)
    field_array[1, 1] = invalid_value

    with pytest.raises(ValidationError):
        grid.flatten(field_array)