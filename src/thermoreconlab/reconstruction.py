"""Forward and inverse heat-source reconstruction methods.

This module contains the steady-state forward heat solver and the
identity-regularized Tikhonov inverse solver used by ThermoReconLab.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix, lil_matrix, vstack
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


def _validate_regularization(
    regularization: object,
) -> Literal["identity", "smoothness"]:
    """Validate the Tikhonov regularization operator selection."""
    if not isinstance(regularization, str):
        raise ValidationError(
            "regularization must be 'identity' or 'smoothness'."
        )

    if regularization not in {"identity", "smoothness"}:
        raise ValidationError(
            "regularization must be 'identity' or 'smoothness'."
        )

    return regularization


def _validate_beta(beta: Real) -> float:
    """Validate a finite nonnegative compactness parameter."""
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise ValidationError("beta must be a real number.")

    beta_value = float(beta)

    if not np.isfinite(beta_value):
        raise ValidationError("beta must be finite.")

    if beta_value < 0.0:
        raise ValidationError("beta must be nonnegative.")

    return beta_value


def _validate_max_iterations(max_iterations: Integral) -> int:
    """Validate a positive iteration limit."""
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, Integral)
    ):
        raise ValidationError(
            "max_iterations must be a positive integer."
        )

    max_iterations_value = int(max_iterations)

    if max_iterations_value <= 0:
        raise ValidationError(
            "max_iterations must be a positive integer."
        )

    return max_iterations_value


def _validate_tolerance(tolerance: Real) -> float:
    """Validate a positive finite convergence tolerance."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise ValidationError("tolerance must be a real number.")

    tolerance_value = float(tolerance)

    if not np.isfinite(tolerance_value):
        raise ValidationError("tolerance must be finite.")

    if tolerance_value <= 0.0:
        raise ValidationError("tolerance must be greater than zero.")

    return tolerance_value


def _build_interior_gradient_matrix(
    grid: Grid2D,
) -> csr_matrix:
    """Build first-order differences between interior source nodes."""
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    interior_nx = grid.nx - 2
    interior_ny = grid.ny - 2
    number_of_unknowns = interior_nx * interior_ny
    number_of_rows = (
        (interior_nx - 1) * interior_ny
        + interior_nx * (interior_ny - 1)
    )

    matrix = lil_matrix(
        (number_of_rows, number_of_unknowns),
        dtype=float,
    )

    def local_index(i: int, j: int) -> int:
        return i * interior_ny + j

    row = 0

    for i in range(interior_nx - 1):
        for j in range(interior_ny):
            matrix[row, local_index(i, j)] = -1.0 / grid.dx
            matrix[row, local_index(i + 1, j)] = 1.0 / grid.dx
            row += 1

    for i in range(interior_nx):
        for j in range(interior_ny - 1):
            matrix[row, local_index(i, j)] = -1.0 / grid.dy
            matrix[row, local_index(i, j + 1)] = 1.0 / grid.dy
            row += 1

    return matrix.tocsr()


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
    *,
    regularization: Literal["identity", "smoothness"] = "identity",
) -> ReconstructionResult:
    """Reconstruct the interior source with Tikhonov regularization.

    Identity regularization minimizes
    ``||Hq-y||^2 + alpha||q||^2`` using the existing dual formula.
    Smoothness regularization minimizes
    ``||Hq-y||^2 + alpha||Lq||^2`` through an augmented least-squares
    system. ``L`` contains first differences scaled by ``grid.dx`` and
    ``grid.dy``, so alpha has units consistent with physical spatial
    derivatives.
    """
    if not isinstance(sensor_data, SensorData):
        raise ValidationError(
            "sensor_data must be a SensorData object."
        )

    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    alpha_value = _validate_alpha(alpha)
    regularization_value = _validate_regularization(regularization)
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

    if regularization_value == "identity":
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
    else:
        gradient_matrix = _build_interior_gradient_matrix(grid)
        augmented_matrix = vstack(
            (
                csr_matrix(observation_matrix),
                np.sqrt(alpha_value) * gradient_matrix,
            ),
            format="csr",
        )
        augmented_values = np.concatenate(
            (
                measurements,
                np.zeros(gradient_matrix.shape[0], dtype=float),
            )
        )

        try:
            optimization = lsq_linear(
                augmented_matrix,
                augmented_values,
                bounds=(-np.inf, np.inf),
                method="trf",
                tol=1e-10,
                lsmr_tol=1e-10,
            )
        except (RuntimeError, ValueError) as error:
            raise SolverError(
                "The smoothness Tikhonov system could not be solved."
            ) from error

        if not optimization.success:
            raise SolverError(
                "The smoothness Tikhonov solver did not converge."
            )

        interior_source = np.asarray(
            optimization.x,
            dtype=float,
        )

    predicted_measurements = (
        observation_matrix @ interior_source
    )

    if not (
        np.all(np.isfinite(interior_source))
        and np.all(np.isfinite(predicted_measurements))
    ):
        raise SolverError(
            "The Tikhonov solver produced non-finite values."
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


def reconstruct_smooth_tikhonov(
    sensor_data: SensorData,
    grid: Grid2D,
    alpha: Real = 1e-9,
    *,
    nonnegative: bool = True,
) -> ReconstructionResult:
    """Reconstruct a smooth source, optionally constrained nonnegative.

    The solver minimizes

    ``||Hq - y||² + alpha ||Gq||²``,

    where ``G`` contains first-order spatial differences between
    neighboring interior source nodes. When ``nonnegative`` is true,
    the additional physical constraint ``q >= 0`` is imposed.
    """
    if not isinstance(sensor_data, SensorData):
        raise ValidationError(
            "sensor_data must be a SensorData object."
        )

    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    if not isinstance(nonnegative, bool):
        raise ValidationError("nonnegative must be a boolean.")

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
    gradient_matrix = _build_interior_gradient_matrix(grid)
    measurements = sensor_data.values.astype(float, copy=True)

    augmented_matrix = vstack(
        (
            csr_matrix(observation_matrix),
            np.sqrt(alpha_value) * gradient_matrix,
        ),
        format="csr",
    )
    augmented_values = np.concatenate(
        (
            measurements,
            np.zeros(gradient_matrix.shape[0], dtype=float),
        )
    )

    bounds = (
        (0.0, np.inf)
        if nonnegative
        else (-np.inf, np.inf)
    )

    try:
        optimization = lsq_linear(
            augmented_matrix,
            augmented_values,
            bounds=bounds,
            method="trf",
            tol=1e-10,
            lsmr_tol=1e-10,
        )
    except (RuntimeError, ValueError) as error:
        raise SolverError(
            "The smooth Tikhonov system could not be solved."
        ) from error

    if not optimization.success:
        raise SolverError(
            "The smooth Tikhonov solver did not converge."
        )

    interior_source = np.asarray(
        optimization.x,
        dtype=float,
    )
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
        n_sensors=len(measurements),
    )


def reconstruct_compact_nonnegative(
    sensor_data: SensorData,
    grid: Grid2D,
    alpha: Real = 1e-9,
    beta: Real = 1e-6,
    *,
    max_iterations: Integral = 20_000,
    tolerance: Real = 1e-8,
) -> ReconstructionResult:
    """Reconstruct a compact, smooth, nonnegative interior source.

    Deterministic FISTA minimizes

    ``0.5||Hq-y||^2 + 0.5 alpha||Gq||^2 + beta||q||_1``

    subject to ``q >= 0``. The defaults are conservative starting
    values for small deterministic problems, not universally optimal
    regularization parameters.
    """
    if not isinstance(sensor_data, SensorData):
        raise ValidationError(
            "sensor_data must be a SensorData object."
        )

    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    alpha_value = _validate_alpha(alpha)
    beta_value = _validate_beta(beta)
    max_iterations_value = _validate_max_iterations(
        max_iterations
    )
    tolerance_value = _validate_tolerance(tolerance)
    validated_indices = custom_sensors(
        sensor_data.indices,
        grid,
    )

    start_time = perf_counter()

    observation_matrix = build_observation_matrix(
        validated_indices,
        grid,
    )
    gradient_matrix = _build_interior_gradient_matrix(grid)
    measurements = sensor_data.values.astype(float, copy=True)

    observation_norm = np.linalg.norm(
        observation_matrix,
        ord=2,
    )
    lipschitz_bound = (
        observation_norm**2
        + alpha_value
        * (4.0 / grid.dx**2 + 4.0 / grid.dy**2)
    )

    if not np.isfinite(lipschitz_bound) or lipschitz_bound <= 0.0:
        raise SolverError(
            "The compact solver produced an invalid step size."
        )

    step_size = 1.0 / lipschitz_bound
    number_of_unknowns = observation_matrix.shape[1]
    interior_source = np.zeros(number_of_unknowns, dtype=float)
    extrapolated_source = interior_source.copy()
    acceleration = 1.0

    for _ in range(max_iterations_value):
        data_gradient = observation_matrix.T @ (
            observation_matrix @ extrapolated_source
            - measurements
        )
        smoothness_gradient = gradient_matrix.T @ (
            gradient_matrix @ extrapolated_source
        )
        gradient = (
            data_gradient
            + alpha_value * smoothness_gradient
        )

        if not np.all(np.isfinite(gradient)):
            raise SolverError(
                "The compact solver produced a non-finite gradient."
            )

        new_source = np.maximum(
            extrapolated_source
            - step_size * gradient
            - step_size * beta_value,
            0.0,
        )

        if not np.all(np.isfinite(new_source)):
            raise SolverError(
                "The compact solver produced non-finite iterates."
            )

        iterate_change = np.linalg.norm(
            new_source - interior_source
        )
        previous_norm = np.linalg.norm(interior_source)

        if iterate_change <= (
            tolerance_value * max(1.0, previous_norm)
        ):
            interior_source = new_source
            break

        new_acceleration = (
            1.0 + np.sqrt(1.0 + 4.0 * acceleration**2)
        ) / 2.0
        extrapolated_source = new_source + (
            (acceleration - 1.0) / new_acceleration
        ) * (new_source - interior_source)

        if not (
            np.isfinite(new_acceleration)
            and np.all(np.isfinite(extrapolated_source))
        ):
            raise SolverError(
                "The compact solver produced non-finite iterates."
            )

        interior_source = new_source
        acceleration = new_acceleration
    else:
        raise SolverError(
            "The compact nonnegative solver did not converge within "
            f"{max_iterations_value} iterations."
        )

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

    residual_norm = float(
        np.linalg.norm(predicted_measurements - measurements)
    )
    solution_norm = float(np.linalg.norm(interior_source))
    runtime = float(perf_counter() - start_time)

    if not (
        np.all(np.isfinite(reconstructed_source))
        and np.all(np.isfinite(predicted_measurements))
        and np.isfinite(residual_norm)
        and np.isfinite(solution_norm)
        and np.isfinite(runtime)
    ):
        raise SolverError(
            "The compact solver produced non-finite outputs."
        )

    return ReconstructionResult(
        source=reconstructed_source,
        predicted_measurements=predicted_measurements.copy(),
        residual_norm=residual_norm,
        solution_norm=solution_norm,
        alpha=alpha_value,
        runtime=runtime,
        n_sensors=len(measurements),
    )
