"""Tests for the finite-difference operators."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.core.operators import (
    build_laplacian_2d,
    build_poisson_matrix,
    flatten_index,
    is_boundary_node,
)
from thermoreconlab.exceptions import ValidationError


def test_flatten_index_matches_c_order() -> None:
    """The index mapping should match NumPy C-order flattening."""
    grid = Grid2D(nx=4, ny=5)

    assert flatten_index(0, 0, grid) == 0
    assert flatten_index(0, 4, grid) == 4
    assert flatten_index(1, 0, grid) == 5
    assert flatten_index(3, 4, grid) == 19


@pytest.mark.parametrize(
    "i, j",
    [
        (-1, 0),
        (4, 0),
        (0, -1),
        (0, 5),
    ],
)
def test_flatten_index_rejects_out_of_range_indices(
    i: int,
    j: int,
) -> None:
    """Grid indices must remain inside the grid."""
    grid = Grid2D(nx=4, ny=5)

    with pytest.raises(ValidationError):
        flatten_index(i, j, grid)


def test_boundary_node_detection() -> None:
    """Boundary and interior points should be identified correctly."""
    grid = Grid2D(nx=5, ny=6)

    assert is_boundary_node(0, 2, grid)
    assert is_boundary_node(4, 2, grid)
    assert is_boundary_node(2, 0, grid)
    assert is_boundary_node(2, 5, grid)

    assert not is_boundary_node(2, 3, grid)


def test_poisson_matrix_has_correct_shape_and_type() -> None:
    """The operator should be a CSR matrix of the expected size."""
    grid = Grid2D(nx=5, ny=6)

    matrix = build_poisson_matrix(grid)

    assert isinstance(matrix, csr_matrix)
    assert matrix.shape == (grid.size, grid.size)


def test_poisson_matrix_contains_finite_values() -> None:
    """All stored sparse matrix values should be finite."""
    grid = Grid2D(nx=5, ny=6)

    matrix = build_poisson_matrix(grid)

    assert np.all(np.isfinite(matrix.data))


def test_boundary_rows_are_identity_rows() -> None:
    """Dirichlet boundary nodes should have identity matrix rows."""
    grid = Grid2D(nx=5, ny=5)
    matrix = build_poisson_matrix(grid)

    boundary_row_index = flatten_index(0, 2, grid)
    row = matrix.getrow(boundary_row_index).toarray().ravel()

    expected = np.zeros(grid.size)
    expected[boundary_row_index] = 1.0

    assert np.array_equal(row, expected)


def test_interior_row_uses_five_point_stencil() -> None:
    """Interior rows should contain the correct stencil coefficients."""
    grid = Grid2D(nx=5, ny=5)
    matrix = build_poisson_matrix(grid)

    i = 2
    j = 2
    row_index = flatten_index(i, j, grid)
    row = matrix.getrow(row_index).toarray().ravel()

    inverse_dx_squared = 1.0 / grid.dx**2
    inverse_dy_squared = 1.0 / grid.dy**2

    expected = np.zeros(grid.size)

    expected[row_index] = (
        2.0 * inverse_dx_squared
        + 2.0 * inverse_dy_squared
    )
    expected[flatten_index(i - 1, j, grid)] = (
        -inverse_dx_squared
    )
    expected[flatten_index(i + 1, j, grid)] = (
        -inverse_dx_squared
    )
    expected[flatten_index(i, j - 1, grid)] = (
        -inverse_dy_squared
    )
    expected[flatten_index(i, j + 1, grid)] = (
        -inverse_dy_squared
    )

    assert np.allclose(row, expected)


def test_interior_row_has_five_nonzero_entries() -> None:
    """Each interior row should contain the five-point stencil."""
    grid = Grid2D(nx=6, ny=6)
    matrix = build_poisson_matrix(grid)

    row_index = flatten_index(3, 3, grid)
    row = matrix.getrow(row_index)

    assert row.nnz == 5


def test_laplacian_function_matches_poisson_matrix() -> None:
    """Both public operator names should return the same matrix."""
    grid = Grid2D(nx=5, ny=6)

    poisson_matrix = build_poisson_matrix(grid)
    laplacian_matrix = build_laplacian_2d(grid)

    difference = poisson_matrix - laplacian_matrix

    assert difference.nnz == 0


def test_invalid_grid_raises_error() -> None:
    """Operator construction requires a Grid2D object."""
    with pytest.raises(ValidationError):
        build_poisson_matrix("invalid grid")  # type: ignore[arg-type]