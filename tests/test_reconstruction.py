"""Tests for the forward and inverse reconstruction methods."""

import numpy as np
import pytest

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.core.operators import build_poisson_matrix
from thermoreconlab.data import gaussian_source
from thermoreconlab.exceptions import SolverError, ValidationError
from thermoreconlab.reconstruction import (
    ReconstructionResult,
    build_observation_matrix,
    reconstruct_compact_nonnegative,
    reconstruct_smooth_tikhonov,
    reconstruct_tikhonov,
    solve_forward,
)
from thermoreconlab.sensors import (
    SensorData,
    create_sensor_data,
    regular_grid_sensors,
    sample_field,
)


def test_forward_solver_returns_correct_shape() -> None:
    grid = Grid2D(nx=10, ny=12)
    source = np.ones(grid.shape)

    temperature = solve_forward(source, grid)

    assert temperature.shape == grid.shape
    assert temperature.dtype == np.float64


def test_forward_solver_returns_finite_values() -> None:
    grid = Grid2D(nx=10, ny=10)
    temperature = solve_forward(np.ones(grid.shape), grid)

    assert np.all(np.isfinite(temperature))


def test_zero_source_returns_zero_temperature() -> None:
    grid = Grid2D(nx=8, ny=9)
    temperature = solve_forward(np.zeros(grid.shape), grid)

    assert np.allclose(temperature, 0.0)


def test_forward_solver_enforces_zero_boundary() -> None:
    grid = Grid2D(nx=10, ny=11)
    temperature = solve_forward(np.ones(grid.shape), grid)

    assert np.allclose(temperature[0, :], 0.0)
    assert np.allclose(temperature[-1, :], 0.0)
    assert np.allclose(temperature[:, 0], 0.0)
    assert np.allclose(temperature[:, -1], 0.0)


def test_positive_source_produces_nonnegative_temperature() -> None:
    grid = Grid2D(nx=12, ny=12)
    temperature = solve_forward(np.ones(grid.shape), grid)

    assert np.all(temperature >= -1e-12)
    assert np.max(temperature[1:-1, 1:-1]) > 0.0


def test_forward_solver_does_not_modify_source() -> None:
    grid = Grid2D(nx=8, ny=8)
    source = np.ones(grid.shape)
    original = source.copy()

    solve_forward(source, grid)

    assert np.array_equal(source, original)


def test_forward_solution_satisfies_linear_system() -> None:
    grid = Grid2D(nx=7, ny=8)
    source = np.ones(grid.shape)
    temperature = solve_forward(source, grid)

    matrix = build_poisson_matrix(grid)
    temperature_vector = temperature.reshape(grid.size, order="C")

    source_matrix = source.copy()
    source_matrix[0, :] = 0.0
    source_matrix[-1, :] = 0.0
    source_matrix[:, 0] = 0.0
    source_matrix[:, -1] = 0.0

    right_hand_side = source_matrix.reshape(grid.size, order="C")
    residual = matrix @ temperature_vector - right_hand_side

    assert np.linalg.norm(residual) < 1e-10


def test_forward_solver_recovers_manufactured_solution() -> None:
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
    grid = Grid2D(nx=8, ny=9)

    with pytest.raises(ValidationError):
        solve_forward(np.ones((9, 8)), grid)


def test_forward_solver_rejects_non_finite_source() -> None:
    grid = Grid2D(nx=8, ny=8)
    source = np.ones(grid.shape)
    source[3, 3] = np.nan

    with pytest.raises(ValidationError):
        solve_forward(source, grid)


def test_forward_solver_rejects_invalid_grid() -> None:
    with pytest.raises(ValidationError):
        solve_forward(
            np.ones((5, 5)),
            "invalid grid",  # type: ignore[arg-type]
        )


def test_observation_matrix_has_expected_shape() -> None:
    grid = Grid2D(nx=7, ny=8)
    sensor_indices = np.array([[1, 1], [2, 4], [5, 6]])

    matrix = build_observation_matrix(sensor_indices, grid)

    assert matrix.shape == (
        3,
        (grid.nx - 2) * (grid.ny - 2),
    )
    assert np.all(np.isfinite(matrix))


def test_boundary_sensor_observation_row_is_zero() -> None:
    grid = Grid2D(nx=5, ny=5)
    matrix = build_observation_matrix(
        np.array([[0, 2], [2, 2]]),
        grid,
    )

    assert np.allclose(matrix[0], 0.0)
    assert not np.allclose(matrix[1], 0.0)


def test_zero_measurements_return_zero_source() -> None:
    grid = Grid2D(nx=7, ny=7)
    indices = regular_grid_sensors(grid, count=9)
    sensor_data = SensorData(
        indices=indices,
        values=np.zeros(len(indices)),
    )

    result = reconstruct_tikhonov(sensor_data, grid, alpha=1e-3)

    assert isinstance(result, ReconstructionResult)
    assert np.allclose(result.source, 0.0)
    assert np.allclose(result.predicted_measurements, 0.0)
    assert result.residual_norm == pytest.approx(0.0)


def test_inverse_result_has_correct_shape_and_diagnostics() -> None:
    grid = Grid2D(nx=8, ny=9)
    source = gaussian_source(
        grid,
        center=(0.5, 0.5),
        sigma=0.12,
    )
    temperature = solve_forward(source, grid)
    indices = regular_grid_sensors(grid, count=12)
    sensor_data = create_sensor_data(temperature, indices, grid)

    result = reconstruct_tikhonov(sensor_data, grid, alpha=1e-4)

    assert isinstance(result, ReconstructionResult)
    assert result.source.shape == grid.shape
    assert result.predicted_measurements.shape == (len(indices),)
    assert np.all(np.isfinite(result.source))
    assert np.isfinite(result.residual_norm)
    assert np.isfinite(result.solution_norm)
    assert result.alpha == pytest.approx(1e-4)
    assert result.runtime >= 0.0
    assert result.n_sensors == len(indices)


def test_reconstructed_source_has_zero_boundary() -> None:
    grid = Grid2D(nx=8, ny=8)
    source = gaussian_source(grid, sigma=0.12)
    temperature = solve_forward(source, grid)
    indices = regular_grid_sensors(grid, count=9)
    sensor_data = create_sensor_data(temperature, indices, grid)

    result = reconstruct_tikhonov(sensor_data, grid, alpha=1e-4)

    assert np.allclose(result.source[0, :], 0.0)
    assert np.allclose(result.source[-1, :], 0.0)
    assert np.allclose(result.source[:, 0], 0.0)
    assert np.allclose(result.source[:, -1], 0.0)


def test_predicted_measurements_match_forward_sampling() -> None:
    grid = Grid2D(nx=8, ny=8)
    source = gaussian_source(grid, sigma=0.12)
    temperature = solve_forward(source, grid)
    indices = regular_grid_sensors(grid, count=9)
    sensor_data = create_sensor_data(temperature, indices, grid)

    result = reconstruct_tikhonov(sensor_data, grid, alpha=1e-4)

    reconstructed_temperature = solve_forward(result.source, grid)
    direct_predictions = sample_field(
        reconstructed_temperature,
        indices,
        grid,
    )

    assert np.allclose(
        result.predicted_measurements,
        direct_predictions,
        atol=1e-10,
    )


def test_inverse_reduces_measurement_residual() -> None:
    grid = Grid2D(nx=9, ny=9)
    source = gaussian_source(
        grid,
        center=(0.45, 0.55),
        sigma=0.12,
    )
    temperature = solve_forward(source, grid)
    indices = regular_grid_sensors(grid, count=16)
    sensor_data = create_sensor_data(temperature, indices, grid)

    result = reconstruct_tikhonov(sensor_data, grid, alpha=1e-6)

    assert result.residual_norm < np.linalg.norm(sensor_data.values)


def test_inverse_reconstruction_is_deterministic() -> None:
    grid = Grid2D(nx=7, ny=7)
    source = gaussian_source(grid, sigma=0.15)
    temperature = solve_forward(source, grid)
    indices = regular_grid_sensors(grid, count=9)
    sensor_data = create_sensor_data(temperature, indices, grid)

    first = reconstruct_tikhonov(sensor_data, grid, alpha=1e-3)
    second = reconstruct_tikhonov(sensor_data, grid, alpha=1e-3)

    assert np.allclose(first.source, second.source)
    assert np.allclose(
        first.predicted_measurements,
        second.predicted_measurements,
    )


@pytest.mark.parametrize(
    "invalid_alpha",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_inverse_rejects_invalid_alpha(
    invalid_alpha: object,
) -> None:
    grid = Grid2D(nx=5, ny=5)
    sensor_data = SensorData(
        indices=np.array([[2, 2]]),
        values=np.array([1.0]),
    )

    with pytest.raises(ValidationError):
        reconstruct_tikhonov(
            sensor_data,
            grid,
            alpha=invalid_alpha,  # type: ignore[arg-type]
        )


def test_inverse_rejects_invalid_sensor_data() -> None:
    grid = Grid2D(nx=5, ny=5)

    with pytest.raises(ValidationError):
        reconstruct_tikhonov(
            "invalid sensor data",  # type: ignore[arg-type]
            grid,
        )

def create_smooth_reconstruction_problem():
    """Create a compact deterministic smooth-source test problem."""
    grid = Grid2D(nx=10, ny=10)
    source = gaussian_source(
        grid,
        center=(0.45, 0.55),
        sigma=0.12,
    )
    temperature = solve_forward(source, grid)
    indices = regular_grid_sensors(grid, count=16)
    sensor_data = create_sensor_data(
        temperature,
        indices,
        grid,
    )

    return grid, source, sensor_data


def test_smooth_solver_returns_nonnegative_result() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    result = reconstruct_smooth_tikhonov(
        sensor_data,
        grid,
        alpha=1e-9,
    )

    assert isinstance(result, ReconstructionResult)
    assert result.source.shape == grid.shape
    assert np.all(np.isfinite(result.source))
    assert np.min(result.source) >= -1e-12
    assert result.runtime >= 0.0


def test_smooth_solver_preserves_zero_boundary() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    result = reconstruct_smooth_tikhonov(
        sensor_data,
        grid,
        alpha=1e-9,
    )

    assert np.allclose(result.source[0, :], 0.0)
    assert np.allclose(result.source[-1, :], 0.0)
    assert np.allclose(result.source[:, 0], 0.0)
    assert np.allclose(result.source[:, -1], 0.0)


def test_smooth_solver_predictions_match_forward_sampling() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    result = reconstruct_smooth_tikhonov(
        sensor_data,
        grid,
        alpha=1e-9,
    )
    reconstructed_temperature = solve_forward(
        result.source,
        grid,
    )
    direct_predictions = sample_field(
        reconstructed_temperature,
        sensor_data.indices,
        grid,
    )

    assert np.allclose(
        result.predicted_measurements,
        direct_predictions,
        atol=1e-10,
    )


def test_smooth_solver_is_deterministic() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    first = reconstruct_smooth_tikhonov(
        sensor_data,
        grid,
        alpha=1e-9,
    )
    second = reconstruct_smooth_tikhonov(
        sensor_data,
        grid,
        alpha=1e-9,
    )

    assert np.allclose(first.source, second.source)
    assert np.allclose(
        first.predicted_measurements,
        second.predicted_measurements,
    )


def test_smooth_solver_zero_measurements_return_zero_source() -> None:
    grid = Grid2D(nx=7, ny=7)
    indices = regular_grid_sensors(grid, count=9)
    sensor_data = SensorData(
        indices=indices,
        values=np.zeros(len(indices)),
    )

    result = reconstruct_smooth_tikhonov(
        sensor_data,
        grid,
        alpha=1e-9,
    )

    assert np.allclose(result.source, 0.0)
    assert np.allclose(result.predicted_measurements, 0.0)
    assert result.residual_norm == pytest.approx(0.0)


def test_smooth_solver_improves_gaussian_source_error() -> None:
    grid, true_source, sensor_data = (
        create_smooth_reconstruction_problem()
    )

    identity_result = reconstruct_tikhonov(
        sensor_data,
        grid,
        alpha=1e-7,
    )
    smooth_result = reconstruct_smooth_tikhonov(
        sensor_data,
        grid,
        alpha=1e-9,
    )

    true_interior = true_source[1:-1, 1:-1]
    identity_error = np.linalg.norm(
        identity_result.source[1:-1, 1:-1] - true_interior
    ) / np.linalg.norm(true_interior)
    smooth_error = np.linalg.norm(
        smooth_result.source[1:-1, 1:-1] - true_interior
    ) / np.linalg.norm(true_interior)

    assert smooth_error < identity_error


@pytest.mark.parametrize(
    "invalid_alpha",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_smooth_solver_rejects_invalid_alpha(
    invalid_alpha: object,
) -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    with pytest.raises(ValidationError):
        reconstruct_smooth_tikhonov(
            sensor_data,
            grid,
            alpha=invalid_alpha,  # type: ignore[arg-type]
        )


def test_smooth_solver_rejects_invalid_nonnegative_flag() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    with pytest.raises(ValidationError):
        reconstruct_smooth_tikhonov(
            sensor_data,
            grid,
            nonnegative=1,  # type: ignore[arg-type]
        )


def test_compact_solver_result_shape_and_diagnostics() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    result = reconstruct_compact_nonnegative(
        sensor_data,
        grid,
        alpha=1e-9,
        beta=1e-6,
    )

    assert isinstance(result, ReconstructionResult)
    assert result.source.shape == grid.shape
    assert result.predicted_measurements.shape == (
        len(sensor_data.values),
    )
    assert np.all(np.isfinite(result.source))
    assert np.all(np.isfinite(result.predicted_measurements))
    assert np.isfinite(result.residual_norm)
    assert np.isfinite(result.solution_norm)
    assert np.isfinite(result.runtime)
    assert result.runtime >= 0.0
    assert result.alpha == pytest.approx(1e-9)
    assert result.n_sensors == len(sensor_data.values)


def test_compact_solver_is_nonnegative_with_zero_boundary() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    result = reconstruct_compact_nonnegative(
        sensor_data,
        grid,
        alpha=1e-9,
        beta=1e-6,
    )

    assert np.all(result.source >= 0.0)
    assert np.allclose(result.source[0, :], 0.0)
    assert np.allclose(result.source[-1, :], 0.0)
    assert np.allclose(result.source[:, 0], 0.0)
    assert np.allclose(result.source[:, -1], 0.0)


def test_compact_solver_is_deterministic() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    first = reconstruct_compact_nonnegative(sensor_data, grid)
    second = reconstruct_compact_nonnegative(sensor_data, grid)

    assert np.array_equal(first.source, second.source)
    assert np.array_equal(
        first.predicted_measurements,
        second.predicted_measurements,
    )


def test_compact_solver_zero_measurements_return_zero_source() -> None:
    grid = Grid2D(nx=7, ny=7)
    indices = regular_grid_sensors(grid, count=9)
    sensor_data = SensorData(
        indices=indices,
        values=np.zeros(len(indices)),
    )

    result = reconstruct_compact_nonnegative(sensor_data, grid)

    assert np.array_equal(result.source, np.zeros(grid.shape))
    assert np.array_equal(
        result.predicted_measurements,
        np.zeros(len(indices)),
    )
    assert result.residual_norm == pytest.approx(0.0)


def test_compact_solver_predictions_match_forward_sampling() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    result = reconstruct_compact_nonnegative(sensor_data, grid)
    reconstructed_temperature = solve_forward(result.source, grid)
    direct_predictions = sample_field(
        reconstructed_temperature,
        sensor_data.indices,
        grid,
    )

    assert np.allclose(
        result.predicted_measurements,
        direct_predictions,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "invalid_alpha",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_compact_solver_rejects_invalid_alpha(
    invalid_alpha: object,
) -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    with pytest.raises(ValidationError):
        reconstruct_compact_nonnegative(
            sensor_data,
            grid,
            alpha=invalid_alpha,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_beta",
    [
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
    ],
)
def test_compact_solver_rejects_invalid_beta(
    invalid_beta: object,
) -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    with pytest.raises(ValidationError):
        reconstruct_compact_nonnegative(
            sensor_data,
            grid,
            beta=invalid_beta,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_max_iterations",
    [0, -1, 1.5, True],
)
def test_compact_solver_rejects_invalid_max_iterations(
    invalid_max_iterations: object,
) -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    with pytest.raises(ValidationError):
        reconstruct_compact_nonnegative(
            sensor_data,
            grid,
            max_iterations=invalid_max_iterations,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_tolerance",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_compact_solver_rejects_invalid_tolerance(
    invalid_tolerance: object,
) -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    with pytest.raises(ValidationError):
        reconstruct_compact_nonnegative(
            sensor_data,
            grid,
            tolerance=invalid_tolerance,  # type: ignore[arg-type]
        )


def test_compact_solver_rejects_invalid_public_objects() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    with pytest.raises(ValidationError):
        reconstruct_compact_nonnegative(
            "invalid",  # type: ignore[arg-type]
            grid,
        )

    with pytest.raises(ValidationError):
        reconstruct_compact_nonnegative(
            sensor_data,
            "invalid",  # type: ignore[arg-type]
        )


def test_compact_solver_reports_convergence_failure() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()

    with pytest.raises(
        SolverError,
        match="did not converge within 1 iterations",
    ):
        reconstruct_compact_nonnegative(
            sensor_data,
            grid,
            alpha=1e-9,
            beta=0.0,
            max_iterations=1,
            tolerance=1e-14,
        )


def test_positive_beta_increases_near_zero_interior_values() -> None:
    grid, _, sensor_data = create_smooth_reconstruction_problem()
    near_zero_threshold = 1e-8

    unpenalized = reconstruct_compact_nonnegative(
        sensor_data,
        grid,
        alpha=1e-9,
        beta=0.0,
    )
    compact = reconstruct_compact_nonnegative(
        sensor_data,
        grid,
        alpha=1e-9,
        beta=1e-6,
    )

    unpenalized_near_zero = np.count_nonzero(
        unpenalized.source[1:-1, 1:-1]
        <= near_zero_threshold
    )
    compact_near_zero = np.count_nonzero(
        compact.source[1:-1, 1:-1]
        <= near_zero_threshold
    )

    assert compact_near_zero >= unpenalized_near_zero
