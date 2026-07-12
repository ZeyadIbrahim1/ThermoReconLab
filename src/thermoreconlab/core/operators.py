"""Finite-difference operators for ThermoReconLab."""

from scipy.sparse import csr_matrix, lil_matrix

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.exceptions import ValidationError


def flatten_index(i: int, j: int, grid: Grid2D) -> int:
    """Convert a two-dimensional grid index to a vector index.

    The package uses C-order flattening, so the mapping is

    ``k = i * ny + j``.

    Parameters
    ----------
    i:
        Grid index in the x-direction.
    j:
        Grid index in the y-direction.
    grid:
        Structured grid defining the index range.

    Returns
    -------
    int
        Corresponding one-dimensional vector index.

    Raises
    ------
    ValidationError
        If the grid or indices are invalid.
    """
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    if not isinstance(i, int) or isinstance(i, bool):
        raise ValidationError("i must be an integer.")

    if not isinstance(j, int) or isinstance(j, bool):
        raise ValidationError("j must be an integer.")

    if not 0 <= i < grid.nx:
        raise ValidationError(
            f"i must satisfy 0 <= i < {grid.nx}."
        )

    if not 0 <= j < grid.ny:
        raise ValidationError(
            f"j must satisfy 0 <= j < {grid.ny}."
        )

    return i * grid.ny + j


def is_boundary_node(i: int, j: int, grid: Grid2D) -> bool:
    """Return whether a grid point lies on the domain boundary.

    Parameters
    ----------
    i:
        Grid index in the x-direction.
    j:
        Grid index in the y-direction.
    grid:
        Structured two-dimensional grid.

    Returns
    -------
    bool
        ``True`` when the point lies on any domain boundary.
    """
    flatten_index(i, j, grid)

    return (
        i == 0
        or i == grid.nx - 1
        or j == 0
        or j == grid.ny - 1
    )


def build_poisson_matrix(grid: Grid2D) -> csr_matrix:
    """Build the sparse finite-difference matrix for ``-ΔT = q``.

    A five-point stencil is used at interior grid points. Boundary
    rows are replaced by identity rows to impose Dirichlet boundary
    values directly.

    For an interior point, the discretization is

    ``(2/dx² + 2/dy²) T[i,j]``
    ``- T[i-1,j]/dx² - T[i+1,j]/dx²``
    ``- T[i,j-1]/dy² - T[i,j+1]/dy²``.

    Parameters
    ----------
    grid:
        Structured two-dimensional grid.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix with shape ``(grid.size, grid.size)``.

    Raises
    ------
    ValidationError
        If ``grid`` is not a Grid2D object.
    """
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    matrix = lil_matrix(
        (grid.size, grid.size),
        dtype=float,
    )

    inverse_dx_squared = 1.0 / grid.dx**2
    inverse_dy_squared = 1.0 / grid.dy**2

    for i in range(grid.nx):
        for j in range(grid.ny):
            row = flatten_index(i, j, grid)

            # Boundary values are imposed directly through identity rows.
            if is_boundary_node(i, j, grid):
                matrix[row, row] = 1.0
                continue

            matrix[row, row] = (
                2.0 * inverse_dx_squared
                + 2.0 * inverse_dy_squared
            )

            matrix[row, flatten_index(i - 1, j, grid)] = (
                -inverse_dx_squared
            )
            matrix[row, flatten_index(i + 1, j, grid)] = (
                -inverse_dx_squared
            )
            matrix[row, flatten_index(i, j - 1, grid)] = (
                -inverse_dy_squared
            )
            matrix[row, flatten_index(i, j + 1, grid)] = (
                -inverse_dy_squared
            )

    return matrix.tocsr()


def build_laplacian_2d(grid: Grid2D) -> csr_matrix:
    """Build the package operator representing the negative Laplacian.

    This function is an explicit public name for the matrix used in
    the heat equation ``-ΔT = q``.

    Parameters
    ----------
    grid:
        Structured two-dimensional grid.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse negative-Laplacian matrix with Dirichlet boundary rows.
    """
    return build_poisson_matrix(grid)