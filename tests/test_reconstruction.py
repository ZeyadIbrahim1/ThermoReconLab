"""Tests for the forward heat solver."""

import numpy as np
import pytest

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.core.operators import build_poisson_matrix
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.reconstruction import solve_forward


def test_forward_solver_returns_correct_shape() -> None:
    """The temperature field should match the source grid."""
    grid = Grid2D(nx=10, ny=12)
    source = np.ones(grid.shape)

    temperature = solve_forward(source, grid)

    assert temperature.shape == grid.shape
    assert temperature.dtype == np.float64


def test_forward_solver_returns_finite_values() -> None:
    """The computed temperature field should contain finite values."""
    grid = Grid2D(nx=10, ny=10)
    source = np.ones(grid.shape)

    temperature = solve_forward(source, grid)

    assert np.all(np.isfinite(temperature))


def test_zero_source_returns_zero_temperature() -> None:
    """A zero source with zero boundary values should give zero."""
    grid = Grid2D(nx=8, ny=9)
    source = np.zeros(grid.shape)

    temperature = solve_forward(source, grid)

    assert np.allclose(temperature, 0.0)


def test_forward_solver_enforces_zero_boundary() -> None:
    """All boundary temperature values should be zero."""
    grid = Grid2D(nx=10, ny=11)
    source = np.ones(grid.shape)

    temperature = solve_forward(source, grid)

    assert np.allclose(temperature[0, :], 0.0)
    assert np.allclose(temperature[-1, :], 0.0)
    assert np.allclose(temperature[:, 0], 0.0)
    assert np.allclose(temperature[:, -1], 0.0)


def test_positive_source_produces_nonnegative_temperature() -> None:
    """A positive source should produce nonnegative temperatures."""
    grid = Grid2D(nx=12, ny=12)
    source = np.ones(grid.shape)

    temperature = solve_forward(source, grid)

    assert np.all(temperature >= -1e-12)
    assert np.max(temperature[1:-1, 1:-1]) > 0.0


def test_forward_solver_does_not_modify_source() -> None:
    """The input source array should remain unchanged."""
    grid = Grid2D(nx=8, ny=8)
    source = np.ones(grid.shape)
    original = source.copy()

    solve_forward(source, grid)

    assert np.array_equal(source, original)


def test_forward_solution_satisfies_linear_system() -> None:
    """The numerical solution should satisfy the discrete equation."""
    grid = Grid2D(nx=7, ny=8)
    source = np.ones(grid.shape)

    temperature = solve_forward(source, grid)

    matrix = build_poisson_matrix(grid)
    temperature_vector = temperature.reshape(grid.size, order="C")

    right_hand_side = source.reshape(grid.size, order="C").copy()

    source_matrix = source.copy()
    source_matrix[0, :] = 0.0
    source_matrix[-1, :] = 0.0
    source_matrix[:, 0] = 0.0
    source_matrix[:, -1] = 0.0

    right_hand_side = source_matrix.reshape(grid.size, order="C")

    residual = matrix @ temperature_vector - right_hand_side

    assert np.linalg.norm(residual) < 1e-10


def test_forward_solver_recovers_manufactured_solution() -> None:
    """A polynomial solution should be reproduced accurately.

    We choose

        T(x, y) = x(1-x)y(1-y)

    which satisfies zero boundary conditions. Its exact source is

        q(x, y) = 2[x(1-x) + y(1-y)].

    The second-order finite-difference stencil is exact for the
    quadratic terms in this manufactured example.
    """
    grid = Grid2D(nx=11, ny=13)

    expected_temperature = (
        grid.X
        * (1.0 - grid.X)
        * grid.Y
        * (1.0 - grid.Y)
    )

    source = 2.0 * (
        grid.X * (1.0 - grid.X)
        + grid.Y * (1.0 - grid.Y)
    )

    computed_temperature = solve_forward(source, grid)

    assert np.allclose(
        computed_temperature,
        expected_temperature,
        atol=1e-12,
    )


def test_forward_solver_rejects_wrong_source_shape() -> None:
    """The source shape must match the grid."""
    grid = Grid2D(nx=8, ny=9)
    source = np.ones((9, 8))

    with pytest.raises(ValidationError):
        solve_forward(source, grid)


def test_forward_solver_rejects_non_finite_source() -> None:
    """Sources containing NaN should be rejected."""
    grid = Grid2D(nx=8, ny=8)
    source = np.ones(grid.shape)
    source[3, 3] = np.nan

    with pytest.raises(ValidationError):
        solve_forward(source, grid)


def test_forward_solver_rejects_invalid_grid() -> None:
    """The grid must be represented by Grid2D."""
    source = np.ones((5, 5))

    with pytest.raises(ValidationError):
        solve_forward(
            source,
            "invalid grid",  # type: ignore[arg-type]
        )