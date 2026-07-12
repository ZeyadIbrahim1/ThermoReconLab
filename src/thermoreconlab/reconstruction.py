"""Forward and inverse heat-source reconstruction methods.

The inverse reconstruction will be added later. For now, this module
contains only the steady-state forward heat solver.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import spsolve

from thermoreconlab.core.fields import (
    flatten_field,
    reshape_field,
    validate_field,
)
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.core.operators import (
    build_poisson_matrix,
    flatten_index,
    is_boundary_node,
)
from thermoreconlab.exceptions import SolverError, ValidationError


def _build_forward_rhs(
    source: NDArray[np.float64],
    grid: Grid2D,
) -> NDArray[np.float64]:
    """Build the right-hand side for homogeneous Dirichlet boundaries.

    The source values are used at interior nodes. Boundary entries are
    replaced by zero because the boundary matrix rows impose ``T = 0``.
    """
    right_hand_side = flatten_field(source, name="source")

    for i in range(grid.nx):
        for j in range(grid.ny):
            if is_boundary_node(i, j, grid):
                index = flatten_index(i, j, grid)
                right_hand_side[index] = 0.0

    return right_hand_side


def solve_forward(
    source: ArrayLike,
    grid: Grid2D,
) -> NDArray[np.float64]:
    """Solve the steady-state heat equation ``-ΔT = q``.

    The equation is discretized using the five-point finite-difference
    operator defined in :mod:`thermoreconlab.core.operators`.
    Homogeneous Dirichlet boundary conditions are imposed:

    ``T = 0`` on the boundary.

    Parameters
    ----------
    source:
        Two-dimensional heat-source field ``q``.
    grid:
        Structured grid on which the problem is solved.

    Returns
    -------
    numpy.ndarray
        Temperature field ``T`` with shape ``grid.shape``.

    Raises
    ------
    ValidationError
        If the grid or source field is invalid.
    SolverError
        If the sparse linear system cannot be solved or produces
        non-finite values.
    """
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    source_array = validate_field(source, grid, name="source")

    poisson_matrix = build_poisson_matrix(grid)
    right_hand_side = _build_forward_rhs(source_array, grid)

    try:
        temperature_vector = spsolve(
            poisson_matrix,
            right_hand_side,
        )
    except (RuntimeError, ValueError) as error:
        raise SolverError(
            "The forward heat equation could not be solved."
        ) from error

    temperature_vector = np.asarray(
        temperature_vector,
        dtype=float,
    )

    if not np.all(np.isfinite(temperature_vector)):
        raise SolverError(
            "The forward solver produced non-finite values."
        )

    return reshape_field(
        temperature_vector,
        grid,
        name="temperature_vector",
    )