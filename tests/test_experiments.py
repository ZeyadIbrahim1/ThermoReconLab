"""Tests for the high-level experiment workflow."""

import numpy as np
import pytest
from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
    run_regularization_study,
)

from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
    reconstruct_from_measurements,
    run_noise_sensitivity_study,
    run_regularization_study,
    run_sensor_count_study,
    run_synthetic_benchmark,
)
from thermoreconlab import run_synthetic_benchmark
from thermoreconlab.experiments import ExperimentResult
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
    reconstruct_from_measurements,
    run_regularization_study,
    run_sensor_count_study,
    run_synthetic_benchmark,
)
from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
)

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.data import gaussian_source
from thermoreconlab.reconstruction import (
    reconstruct_tikhonov,
    solve_forward,
)
from thermoreconlab.sensors import (
    SensorData,
    create_sensor_data,
    regular_grid_sensors,
)

from thermoreconlab import (
    reconstruct_from_measurements,
    run_synthetic_benchmark,
)
def test_synthetic_benchmark_returns_experiment_result() -> None:
    result = run_synthetic_benchmark(
        grid_shape=(9, 10),
        num_sensors=12,
        noise_level=0.01,
        alpha=1e-4,
        seed=42,
    )

    assert isinstance(result, ExperimentResult)
    assert result.grid.shape == (9, 10)
    assert result.true_source.shape == result.grid.shape
    assert result.temperature.shape == result.grid.shape
    assert result.reconstructed_source.shape == result.grid.shape
    assert len(result.sensor_data_clean) == 12
    assert len(result.sensor_data_noisy) == 12
    assert result.runtime >= 0.0


def test_benchmark_contains_expected_metrics() -> None:
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        seed=4,
    )

    assert set(result.metrics) == {
        "rmse",
        "mae",
        "relative_l2_error",
        "max_absolute_error",
        "residual_norm",
        "solution_norm",
    }


def test_benchmark_is_reproducible() -> None:
    settings = {
        "grid_shape": (9, 9),
        "source_type": "random_hotspots",
        "sensor_strategy": "random",
        "num_sensors": 10,
        "noise_level": 0.02,
        "alpha": 1e-4,
        "seed": 17,
    }

    first = run_synthetic_benchmark(**settings)
    second = run_synthetic_benchmark(**settings)

    assert np.array_equal(first.true_source, second.true_source)
    assert np.array_equal(
        first.sensor_data_clean.indices,
        second.sensor_data_clean.indices,
    )
    assert np.array_equal(
        first.sensor_data_noisy.values,
        second.sensor_data_noisy.values,
    )
    assert np.allclose(
        first.reconstructed_source,
        second.reconstructed_source,
    )


def test_zero_noise_preserves_measurements() -> None:
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        noise_level=0.0,
        seed=42,
    )

    assert np.array_equal(
        result.sensor_data_clean.values,
        result.sensor_data_noisy.values,
    )


@pytest.mark.parametrize(
    "source_type",
    ["gaussian", "two_gaussians", "random_hotspots"],
)
def test_supported_source_types_run(
    source_type: str,
) -> None:
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        source_type=source_type,
        num_sensors=9,
        seed=3,
    )

    assert result.true_source.shape == (8, 8)
    assert np.all(result.true_source >= 0.0)


@pytest.mark.parametrize(
    "sensor_strategy",
    ["regular", "random", "center_focused"],
)
def test_supported_sensor_strategies_run(
    sensor_strategy: str,
) -> None:
    result = run_synthetic_benchmark(
        grid_shape=(9, 9),
        sensor_strategy=sensor_strategy,
        num_sensors=9,
        seed=6,
    )

    indices = result.sensor_data_clean.indices

    assert indices.shape == (9, 2)
    assert np.all(indices[:, 0] > 0)
    assert np.all(indices[:, 0] < result.grid.nx - 1)
    assert np.all(indices[:, 1] > 0)
    assert np.all(indices[:, 1] < result.grid.ny - 1)


def test_result_to_dict_returns_compact_summary() -> None:
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        seed=1,
    )

    summary = result.to_dict()

    assert set(summary) == {
        "config",
        "metrics",
        "runtime",
        "reconstruction",
    }
    assert summary["config"]["grid_shape"] == (8, 8)
    assert summary["reconstruction"]["n_sensors"] == 9
    assert "true_source" not in summary


def test_clean_sensor_values_match_temperature_field() -> None:
    result = run_synthetic_benchmark(
        grid_shape=(8, 9),
        num_sensors=10,
        noise_level=0.0,
        seed=2,
    )

    indices = result.sensor_data_clean.indices
    expected = result.temperature[
        indices[:, 0],
        indices[:, 1],
    ]

    assert np.allclose(
        result.sensor_data_clean.values,
        expected,
    )


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (2, 8),
        (8, 2),
        (8,),
        [8, 8],
        (8.5, 8),
        (True, 8),
    ],
)
def test_benchmark_rejects_invalid_grid_shape(
    invalid_shape: object,
) -> None:
    with pytest.raises(ValidationError):
        run_synthetic_benchmark(
            grid_shape=invalid_shape,  # type: ignore[arg-type]
        )


def test_benchmark_rejects_unknown_source_type() -> None:
    with pytest.raises(ValidationError):
        run_synthetic_benchmark(
            grid_shape=(8, 8),
            source_type="square_hotspot",
            num_sensors=9,
        )


def test_benchmark_rejects_unknown_sensor_strategy() -> None:
    with pytest.raises(ValidationError):
        run_synthetic_benchmark(
            grid_shape=(8, 8),
            sensor_strategy="diagonal",
            num_sensors=9,
        )


def test_boundary_only_strategy_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="contain no source information",
    ):
        run_synthetic_benchmark(
            grid_shape=(8, 8),
            sensor_strategy="boundary",
            num_sensors=9,
        )


@pytest.mark.parametrize(
    "invalid_noise",
    [-0.1, float("nan"), float("inf"), True],
)
def test_benchmark_rejects_invalid_noise_level(
    invalid_noise: object,
) -> None:
    with pytest.raises(ValidationError):
        run_synthetic_benchmark(
            grid_shape=(8, 8),
            num_sensors=9,
            noise_level=invalid_noise,  # type: ignore[arg-type]
        )


def create_example_measurements() -> tuple[Grid2D, SensorData]:
    """Create deterministic measurements for user-mode tests."""
    grid = Grid2D(nx=9, ny=9)

    source = gaussian_source(
        grid,
        center=(0.5, 0.5),
        sigma=0.12,
    )

    temperature = solve_forward(source, grid)
    indices = regular_grid_sensors(grid, count=12)

    sensor_data = create_sensor_data(
        temperature,
        indices,
        grid,
    )

    return grid, sensor_data


def test_reconstruct_from_measurements_returns_result() -> None:
    """User measurements should produce a structured result."""
    grid, sensor_data = create_example_measurements()

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
        alpha=1e-4,
    )

    assert isinstance(
        result,
        MeasurementReconstructionResult,
    )
    assert result.grid.shape == grid.shape
    assert result.reconstructed_source.shape == grid.shape
    assert result.reconstruction.n_sensors == len(sensor_data)
    assert result.runtime >= 0.0


def test_user_mode_matches_direct_inverse_solver() -> None:
    """The high-level workflow should use the same inverse solver."""
    grid, sensor_data = create_example_measurements()

    workflow_result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
        alpha=1e-4,
    )

    direct_result = reconstruct_tikhonov(
        sensor_data,
        grid,
        alpha=1e-4,
    )

    assert np.allclose(
        workflow_result.reconstructed_source,
        direct_result.source,
    )

    assert (
        workflow_result.reconstruction.residual_norm
        == pytest.approx(direct_result.residual_norm)
    )


def test_user_result_summary_contains_no_ground_truth_metrics() -> None:
    """Real-data results should not claim unavailable accuracy metrics."""
    grid, sensor_data = create_example_measurements()

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
    )

    summary = result.to_dict()

    assert set(summary) == {
        "config",
        "runtime",
        "reconstruction",
    }

    assert summary["config"]["mode"] == "user_measurements"
    assert summary["config"]["num_sensors"] == len(sensor_data)
    assert "metrics" not in summary
    assert "true_source" not in summary


def test_user_mode_rejects_out_of_range_sensor_indices() -> None:
    """Sensor indices must be valid for the requested grid."""
    sensor_data = SensorData(
        indices=np.array(
            [
                [1, 1],
                [8, 3],
            ]
        ),
        values=np.array([0.1, 0.2]),
    )

    with pytest.raises(ValidationError):
        reconstruct_from_measurements(
            sensor_data,
            grid_shape=(8, 8),
        )


def test_user_mode_rejects_invalid_sensor_data() -> None:
    """The public workflow requires a SensorData object."""
    with pytest.raises(ValidationError):
        reconstruct_from_measurements(
            np.ones((3, 3)),  # type: ignore[arg-type]
            grid_shape=(8, 8),
        )


def test_regularization_study_returns_dataframe() -> None:
    """The study should return one row per alpha value."""
    alpha_values = [1e-4, 1e-5, 1e-6]

    dataframe, results = run_regularization_study(
        alpha_values,
        grid_shape=(8, 8),
        num_sensors=9,
        seed=42,
    )

    assert len(dataframe) == len(alpha_values)
    assert len(results) == len(alpha_values)

    assert set(dataframe.columns) == {
        "study_type",
        "alpha",
        "rmse",
        "mae",
        "relative_l2_error",
        "max_absolute_error",
        "residual_norm",
        "solution_norm",
        "max_reconstructed_source",
        "runtime",
    }


def test_regularization_study_uses_same_measurements() -> None:
    """Only alpha should change between study runs."""
    _, results = run_regularization_study(
        [1e-4, 1e-5],
        grid_shape=(8, 8),
        sensor_strategy="random",
        num_sensors=9,
        seed=17,
    )

    assert np.array_equal(
        results[0].true_source,
        results[1].true_source,
    )

    assert np.array_equal(
        results[0].sensor_data_noisy.indices,
        results[1].sensor_data_noisy.indices,
    )

    assert np.array_equal(
        results[0].sensor_data_noisy.values,
        results[1].sensor_data_noisy.values,
    )


def test_regularization_study_rejects_empty_values() -> None:
    """At least one alpha value must be supplied."""
    with pytest.raises(ValidationError):
        run_regularization_study([])


def test_regularization_study_rejects_invalid_alpha() -> None:
    """Invalid alpha values should be rejected."""
    with pytest.raises(ValidationError):
        run_regularization_study(
            [1e-4, 0.0],
            grid_shape=(8, 8),
            num_sensors=9,
        )       

def test_sensor_count_study_returns_dataframe() -> None:
    """The study should return one result per sensor count."""
    sensor_counts = [4, 9, 16]

    dataframe, results = run_sensor_count_study(
        sensor_counts,
        grid_shape=(10, 10),
        sensor_strategy="random",
        noise_level=0.01,
        seed=42,
    )

    assert len(dataframe) == len(sensor_counts)
    assert len(results) == len(sensor_counts)

    assert dataframe["sensor_count"].tolist() == sensor_counts

    assert set(dataframe.columns) == {
        "study_type",
        "sensor_count",
        "sensor_fraction",
        "rmse",
        "mae",
        "relative_l2_error",
        "max_absolute_error",
        "residual_norm",
        "solution_norm",
        "runtime",
    }


def test_sensor_count_study_uses_same_true_source() -> None:
    """Changing sensor count should not change the true source."""
    _, results = run_sensor_count_study(
        [4, 9],
        grid_shape=(10, 10),
        sensor_strategy="random",
        seed=17,
    )

    assert np.array_equal(
        results[0].true_source,
        results[1].true_source,
    )


def test_sensor_count_study_uses_requested_counts() -> None:
    """Each reconstruction should use its requested sensor count."""
    sensor_counts = [5, 10]

    _, results = run_sensor_count_study(
        sensor_counts,
        grid_shape=(10, 10),
        sensor_strategy="random",
        seed=42,
    )

    actual_counts = [
        result.reconstruction.n_sensors
        for result in results
    ]

    assert actual_counts == sensor_counts


def test_sensor_count_study_rejects_empty_values() -> None:
    """At least one sensor count must be supplied."""
    with pytest.raises(ValidationError):
        run_sensor_count_study([])


@pytest.mark.parametrize(
    "invalid_counts",
    [
        [0, 4],
        [-1, 4],
        [4, 2.5],
        [4, True],
    ],
)
def test_sensor_count_study_rejects_invalid_counts(
    invalid_counts: list[object],
) -> None:
    """Sensor counts must be positive integers."""
    with pytest.raises(ValidationError):
        run_sensor_count_study(
            invalid_counts,  # type: ignore[arg-type]
        )        


def test_noise_sensitivity_study_returns_dataframe() -> None:
    """The study should return one result per noise level."""
    noise_levels = [0.0, 0.01, 0.05]

    dataframe, results = run_noise_sensitivity_study(
        noise_levels,
        grid_shape=(10, 10),
        num_sensors=16,
        seed=42,
    )

    assert len(dataframe) == len(noise_levels)
    assert len(results) == len(noise_levels)

    assert dataframe["noise_level"].tolist() == noise_levels

    assert set(dataframe.columns) == {
        "study_type",
        "noise_level",
        "measurement_noise_norm",
        "mean_absolute_measurement_noise",
        "rmse",
        "mae",
        "relative_l2_error",
        "max_absolute_error",
        "residual_norm",
        "solution_norm",
        "runtime",
    }


def test_noise_study_uses_same_source_and_sensors() -> None:
    """Only the measurement-noise magnitude should change."""
    _, results = run_noise_sensitivity_study(
        [0.0, 0.02],
        grid_shape=(10, 10),
        sensor_strategy="random",
        num_sensors=12,
        seed=17,
    )

    assert np.array_equal(
        results[0].true_source,
        results[1].true_source,
    )

    assert np.array_equal(
        results[0].sensor_data_clean.indices,
        results[1].sensor_data_clean.indices,
    )

    assert np.array_equal(
        results[0].sensor_data_clean.values,
        results[1].sensor_data_clean.values,
    )


def test_noise_study_zero_level_preserves_measurements() -> None:
    """Zero noise should leave the sensor values unchanged."""
    dataframe, results = run_noise_sensitivity_study(
        [0.0],
        grid_shape=(10, 10),
        num_sensors=16,
        seed=42,
    )

    assert np.array_equal(
        results[0].sensor_data_clean.values,
        results[0].sensor_data_noisy.values,
    )

    assert dataframe.loc[0, "measurement_noise_norm"] == pytest.approx(
        0.0
    )


def test_noise_study_rejects_empty_values() -> None:
    """At least one noise level must be supplied."""
    with pytest.raises(ValidationError):
        run_noise_sensitivity_study([])


@pytest.mark.parametrize(
    "invalid_levels",
    [
        [-0.01, 0.02],
        [0.01, float("nan")],
        [0.01, float("inf")],
        [0.01, True],
    ],
)
def test_noise_study_rejects_invalid_levels(
    invalid_levels: list[object],
) -> None:
    """Noise levels must be finite nonnegative numbers."""
    with pytest.raises(ValidationError):
        run_noise_sensitivity_study(
            invalid_levels,  # type: ignore[arg-type]
        )
                