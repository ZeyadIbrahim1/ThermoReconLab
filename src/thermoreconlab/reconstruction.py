"""Forward and inverse heat-source reconstruction methods.

This module contains the steady-state forward heat solver and the
identity-regularized Tikhonov inverse solver used by ThermoReconLab.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import splu, spsolve

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
from thermoreconlab.sensors import SensorData, custom_sensors


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """Store an inverse reconstruction and its diagnostics."""

    source: NDArray[np.float64]
    predicted_measurements: NDArray[np.float64]
    residual_norm: float
    solution_norm: float
    alpha: float
    runtime: float
    n_sensors: int


def _build_forward_rhs(
    source: NDArray[np.float64],
    grid: Grid2D,
) -> NDArray[np.float64]:
    """Build the right-hand side for zero Dirichlet boundaries."""
    right_hand_side = flatten_field(source, name="source")

    for i in range(grid.nx):
        for j in range(grid.ny):
            if is_boundary_node(i, j, grid):
                index = flatten_index(i, j, grid)
                right_hand_side[index] = 0.0

    return right_hand_side


def _interior_flat_indices(grid: Grid2D) -> NDArray[np.int64]:
    """Return flattened indices of all interior grid nodes."""
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    indices = [
        flatten_index(i, j, grid)
        for i in range(1, grid.nx - 1)
        for j in range(1, grid.ny - 1)
    ]

    return np.asarray(indices, dtype=np.int64)


def _validate_alpha(alpha: Real) -> float:
    """Validate a positive finite regularization parameter."""
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValidationError("alpha must be a real number.")

    alpha_value = float(alpha)

    if not np.isfinite(alpha_value):
        raise ValidationError("alpha must be finite.")

    if alpha_value <= 0.0:
        raise ValidationError("alpha must be greater than zero.")

    return alpha_value


def solve_forward(
    source: ArrayLike,
    grid: Grid2D,
) -> NDArray[np.float64]:
    """Solve the steady-state heat equation ``-Delta T = q``."""
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


def build_observation_matrix(
    sensor_indices: ArrayLike,
    grid: Grid2D,
) -> NDArray[np.float64]:
    """Build ``H = S A^-1 E`` for interior source values.

    One adjoint solve is used per sensor, which is efficient when the
    number of sensors is smaller than the number of source unknowns.
    """
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    indices = custom_sensors(sensor_indices, grid)
    n_sensors = len(indices)

    sensor_flat_indices = np.asarray(
        [
            flatten_index(int(i), int(j), grid)
            for i, j in indices
        ],
        dtype=np.int64,
    )

    sensor_right_hand_sides = np.zeros(
        (grid.size, n_sensors),
        dtype=float,
    )
    sensor_right_hand_sides[
        sensor_flat_indices,
        np.arange(n_sensors),
    ] = 1.0

    poisson_matrix = build_poisson_matrix(grid)

    try:
        factorization = splu(
            poisson_matrix.transpose().tocsc()
        )
        adjoint_solutions = factorization.solve(
            sensor_right_hand_sides
        )
    except (RuntimeError, ValueError) as error:
        raise SolverError(
            "The observation matrix could not be constructed."
        ) from error

    adjoint_solutions = np.asarray(
        adjoint_solutions,
        dtype=float,
    )

    if adjoint_solutions.ndim == 1:
        adjoint_solutions = adjoint_solutions[:, np.newaxis]

    interior_indices = _interior_flat_indices(grid)
    observation_matrix = (
        adjoint_solutions[interior_indices, :].T.copy()
    )

    if not np.all(np.isfinite(observation_matrix)):
        raise SolverError(
            "The observation matrix contains non-finite values."
        )

    return observation_matrix


def reconstruct_tikhonov(
    sensor_data: SensorData,
    grid: Grid2D,
    alpha: Real = 1e-3,
) -> ReconstructionResult:
    """Reconstruct the interior source with identity Tikhonov.

    The solver minimizes ``||Hq-y||^2 + alpha||q||^2`` and uses the
    dual formula ``q = H.T solve(H H.T + alpha I, y)``.
    """
    if not isinstance(sensor_data, SensorData):
        raise ValidationError(
            "sensor_data must be a SensorData object."
        )

    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    alpha_value = _validate_alpha(alpha)
    validated_indices = custom_sensors(
        sensor_data.indices,
        grid,
    )

    start_time = perf_counter()

    observation_matrix = build_observation_matrix(
        validated_indices,
        grid,
    )
    measurements = sensor_data.values.astype(float, copy=True)

    n_sensors = len(measurements)
    dual_matrix = (
        observation_matrix @ observation_matrix.T
        + alpha_value * np.eye(n_sensors)
    )

    try:
        dual_weights = np.linalg.solve(
            dual_matrix,
            measurements,
        )
    except np.linalg.LinAlgError as error:
        raise SolverError(
            "The Tikhonov system could not be solved."
        ) from error

    interior_source = observation_matrix.T @ dual_weights
    predicted_measurements = (
        observation_matrix @ interior_source
    )

    full_source_vector = np.zeros(grid.size, dtype=float)
    interior_indices = _interior_flat_indices(grid)
    full_source_vector[interior_indices] = interior_source

    reconstructed_source = reshape_field(
        full_source_vector,
        grid,
        name="reconstructed_source_vector",
    )

    runtime = perf_counter() - start_time

    return ReconstructionResult(
        source=reconstructed_source,
        predicted_measurements=predicted_measurements.copy(),
        residual_norm=float(
            np.linalg.norm(
                predicted_measurements - measurements
            )
        ),
        solution_norm=float(np.linalg.norm(interior_source)),
        alpha=alpha_value,
        runtime=float(runtime),
        n_sensors=n_sensors,
    )
