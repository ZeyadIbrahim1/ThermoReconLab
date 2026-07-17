"""Tests for the high-level experiment workflow."""

import numpy as np
import pandas as pd
import pytest

import thermoreconlab.experiments as experiments_module

from thermoreconlab.core.domain import Domain2D
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.data import gaussian_source, load_sensor_csv
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
    reconstruct_from_measurements,
    reconstruct_from_temperature_field,
    run_compact_parameter_study,
    run_noise_sensitivity_study,
    run_reconstruction_method_study,
    run_regularization_comparison,
    run_regularization_study,
    run_repeated_noise_study,
    run_sensor_layout_study,
    run_sensor_count_study,
    run_synthetic_benchmark,
)
from thermoreconlab.reconstruction import (
    reconstruct_tikhonov,
    solve_forward,
)
from thermoreconlab.sensors import (
    SensorData,
    create_sensor_data,
    regular_grid_sensors,
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
        "relative_residual",
        "residual_rms",
        "solution_norm",
    }


def test_benchmark_residual_metrics_are_finite() -> None:
    """Residual diagnostics should be finite normally."""
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        noise_level=0.02,
        seed=4,
    )

    for name in (
        "residual_norm",
        "relative_residual",
        "residual_rms",
    ):
        assert np.isfinite(result.metrics[name])

    assert result.metrics["residual_norm"] == pytest.approx(
        result.reconstruction.residual_norm
    )


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


def test_user_mode_contains_measurement_metrics_only() -> None:
    """User mode should expose measurement diagnostics only."""
    grid, sensor_data = create_example_measurements()

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
    )

    assert set(result.metrics) == {
        "residual_norm",
        "relative_residual",
        "residual_rms",
        "solution_norm",
    }
    assert all(np.isfinite(value) for value in result.metrics.values())

    for source_metric in (
        "rmse",
        "mae",
        "relative_l2_error",
        "max_absolute_error",
    ):
        assert source_metric not in result.metrics


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


def test_repeated_noise_study_returns_expected_tables() -> None:
    """Each noise level should contain one row per noise seed."""
    detailed, summary, results = run_repeated_noise_study(
        [0.0, 0.02],
        [10, 20, 30],
        grid_shape=(8, 8),
        num_sensors=9,
        seed=42,
    )

    assert len(detailed) == 6
    assert len(summary) == 2
    assert len(results) == 6
    assert summary["number_of_runs"].tolist() == [3, 3]

    assert set(summary.columns) == {
        "noise_level",
        "number_of_runs",
        "mean_relative_l2_error",
        "std_relative_l2_error",
        "mean_rmse",
        "std_rmse",
        "mean_residual_norm",
        "std_residual_norm",
    }

    reference_source = results[0].true_source
    reference_indices = results[0].sensor_data_clean.indices
    reference_values = results[0].sensor_data_clean.values

    for result in results[1:]:
        assert np.array_equal(result.true_source, reference_source)
        assert np.array_equal(
            result.sensor_data_clean.indices,
            reference_indices,
        )
        assert np.array_equal(
            result.sensor_data_clean.values,
            reference_values,
        )


def test_repeated_noise_study_is_deterministic() -> None:
    """Fixed source and noise seeds should reproduce all metrics."""
    settings = {
        "noise_levels": [0.01, 0.03],
        "seeds": [4, 9],
        "grid_shape": (8, 8),
        "sensor_strategy": "random",
        "num_sensors": 9,
        "seed": 17,
    }

    first_detail, first_summary, first_results = (
        run_repeated_noise_study(**settings)
    )
    second_detail, second_summary, second_results = (
        run_repeated_noise_study(**settings)
    )

    assert first_detail.drop(columns="runtime").equals(
        second_detail.drop(columns="runtime")
    )
    assert first_summary.equals(second_summary)

    for first, second in zip(first_results, second_results):
        assert np.array_equal(
            first.sensor_data_noisy.values,
            second.sensor_data_noisy.values,
        )
        assert np.allclose(
            first.reconstructed_source,
            second.reconstructed_source,
        )


def test_repeated_noise_study_one_seed_has_zero_std() -> None:
    """One realization has no observed between-run variability."""
    _, summary, _ = run_repeated_noise_study(
        [0.0, 0.02],
        [10],
        grid_shape=(8, 8),
        num_sensors=9,
    )

    standard_deviations = summary[
        [
            "std_relative_l2_error",
            "std_rmse",
            "std_residual_norm",
        ]
    ].to_numpy()

    assert np.allclose(standard_deviations, 0.0)


def test_repeated_noise_study_rejects_empty_seeds() -> None:
    """At least one noise seed must be supplied."""
    with pytest.raises(ValidationError):
        run_repeated_noise_study([0.01], [])


@pytest.mark.parametrize(
    "invalid_seeds",
    [
        [-1, 2],
        [1.5, 2],
        [True, 2],
        ["bad", 2],
    ],
)
def test_repeated_noise_study_rejects_invalid_seeds(
    invalid_seeds: list[object],
) -> None:
    """Noise seeds must be nonnegative integers."""
    with pytest.raises(ValidationError):
        run_repeated_noise_study(
            [0.01],
            invalid_seeds,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_levels",
    [
        [-0.01, 0.02],
        [0.01, float("nan")],
        [0.01, float("inf")],
        [0.01, True],
    ],
)
def test_repeated_noise_study_rejects_invalid_noise_levels(
    invalid_levels: list[object],
) -> None:
    """Noise levels must be finite nonnegative numbers."""
    with pytest.raises(ValidationError):
        run_repeated_noise_study(
            invalid_levels,  # type: ignore[arg-type]
            [10, 20],
        )


def run_small_sensor_layout_study():
    """Run a compact deterministic sensor-layout comparison."""
    return run_sensor_layout_study(
        ["regular", "random", "center_focused"],
        [1, 2],
        grid_shape=(10, 10),
        num_sensors=9,
        noise_level=0.01,
        alpha=1e-6,
        seed=42,
    )


def test_sensor_layout_study_returns_expected_rows() -> None:
    detailed, summary, results = run_small_sensor_layout_study()

    assert len(detailed) == 6
    assert len(summary) == 3
    assert len(results) == 6


def test_sensor_layout_study_has_required_columns() -> None:
    detailed, summary, _ = run_small_sensor_layout_study()

    assert set(detailed.columns) == {
        "strategy",
        "run_seed",
        "sensor_count",
        "relative_l2_error",
        "rmse",
        "residual_norm",
        "relative_residual",
        "residual_rms",
        "runtime_seconds",
    }
    assert set(summary.columns) == {
        "strategy",
        "number_of_runs",
        "mean_relative_l2_error",
        "std_relative_l2_error",
        "mean_rmse",
        "std_rmse",
        "mean_residual_norm",
        "std_residual_norm",
        "mean_runtime_seconds",
    }


def test_sensor_layout_study_is_scientifically_deterministic() -> None:
    first_detail, first_summary, first_results = (
        run_small_sensor_layout_study()
    )
    second_detail, second_summary, second_results = (
        run_small_sensor_layout_study()
    )

    pd.testing.assert_frame_equal(
        first_detail.drop(columns="runtime_seconds"),
        second_detail.drop(columns="runtime_seconds"),
    )
    pd.testing.assert_frame_equal(
        first_summary.drop(columns="mean_runtime_seconds"),
        second_summary.drop(columns="mean_runtime_seconds"),
    )

    for first, second in zip(first_results, second_results):
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


def test_sensor_layout_study_uses_same_true_source() -> None:
    _, _, results = run_small_sensor_layout_study()
    reference = results[0].true_source

    for result in results[1:]:
        assert np.array_equal(result.true_source, reference)


def test_random_layout_varies_across_run_seeds() -> None:
    _, _, results = run_small_sensor_layout_study()
    random_results = [
        result
        for result in results
        if result.config["sensor_strategy"] == "random"
    ]

    assert not np.array_equal(
        random_results[0].sensor_data_clean.indices,
        random_results[1].sensor_data_clean.indices,
    )


@pytest.mark.parametrize(
    "strategy",
    ["regular", "center_focused"],
)
def test_fixed_layouts_are_reproducible_across_runs(
    strategy: str,
) -> None:
    _, _, results = run_small_sensor_layout_study()
    selected = [
        result
        for result in results
        if result.config["sensor_strategy"] == strategy
    ]

    assert np.array_equal(
        selected[0].sensor_data_clean.indices,
        selected[1].sensor_data_clean.indices,
    )


def test_sensor_layout_study_one_seed_has_zero_std() -> None:
    _, summary, _ = run_sensor_layout_study(
        ["regular", "random", "center_focused"],
        [7],
        grid_shape=(10, 10),
        num_sensors=9,
        noise_level=0.01,
        alpha=1e-6,
        seed=42,
    )

    std_values = summary[
        [
            "std_relative_l2_error",
            "std_rmse",
            "std_residual_norm",
        ]
    ].to_numpy()

    assert np.allclose(std_values, 0.0)


def test_sensor_layout_study_metrics_are_finite() -> None:
    detailed, summary, _ = run_small_sensor_layout_study()

    metric_columns = [
        "relative_l2_error",
        "rmse",
        "residual_norm",
        "relative_residual",
        "residual_rms",
    ]
    assert np.all(
        np.isfinite(detailed[metric_columns].to_numpy())
    )
    assert np.all(
        np.isfinite(
            summary.drop(columns="strategy").to_numpy()
        )
    )
    assert np.all(detailed["runtime_seconds"] >= 0.0)


def test_sensor_layout_study_preserves_strategy_order() -> None:
    _, summary, _ = run_sensor_layout_study(
        ["center_focused", "regular", "random"],
        [1],
        grid_shape=(10, 10),
        num_sensors=9,
        seed=42,
    )

    assert summary["strategy"].tolist() == [
        "center_focused",
        "regular",
        "random",
    ]


def test_sensor_layout_study_rejects_empty_seeds() -> None:
    with pytest.raises(ValidationError):
        run_sensor_layout_study(["regular"], [])


@pytest.mark.parametrize(
    "invalid_seeds",
    [[-1], [1.5], [True], ["bad"]],
)
def test_sensor_layout_study_rejects_invalid_seeds(
    invalid_seeds: list[object],
) -> None:
    with pytest.raises(ValidationError):
        run_sensor_layout_study(
            ["regular"],
            invalid_seeds,  # type: ignore[arg-type]
        )


def test_sensor_layout_study_rejects_empty_strategies() -> None:
    with pytest.raises(ValidationError):
        run_sensor_layout_study([], [1])


def test_sensor_layout_study_rejects_unknown_strategy() -> None:
    with pytest.raises(ValidationError):
        run_sensor_layout_study(["diagonal"], [1])


def test_sensor_layout_study_rejects_duplicate_strategies() -> None:
    with pytest.raises(ValidationError):
        run_sensor_layout_study(
            ["center", "center_focused"],
            [1],
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("num_sensors", 0),
        ("num_sensors", True),
        ("noise_level", -0.01),
        ("noise_level", float("nan")),
        ("alpha", 0.0),
        ("alpha", float("inf")),
        ("seed", -1),
        ("seed", True),
    ],
)
def test_sensor_layout_study_rejects_invalid_parameters(
    keyword: str,
    value: object,
) -> None:
    settings = {
        "strategies": ["regular"],
        "seeds": [1],
        "grid_shape": (10, 10),
        "num_sensors": 9,
        "noise_level": 0.01,
        "alpha": 1e-6,
        "seed": 42,
    }
    settings[keyword] = value

    with pytest.raises(ValidationError):
        run_sensor_layout_study(**settings)  # type: ignore[arg-type]


COMPACT_STUDY_COLUMNS = [
    "alpha",
    "beta",
    "relative_l2_error",
    "rmse",
    "mae",
    "max_absolute_error",
    "residual_norm",
    "relative_residual",
    "residual_rms",
    "solution_norm",
    "gradient_norm",
    "near_zero_count",
    "near_zero_fraction",
    "active_count",
    "runtime_seconds",
]


def run_small_compact_parameter_study() -> pd.DataFrame:
    """Run a small deterministic compact-parameter study."""
    return run_compact_parameter_study(
        [1e-10, 1e-9],
        [0.0, 1e-7],
        grid_shape=(8, 8),
        num_sensors=9,
        noise_level=0.01,
        seed=17,
        near_zero_threshold=1e-8,
    )


def test_compact_parameter_study_shape_columns_and_order() -> None:
    dataframe = run_small_compact_parameter_study()

    assert len(dataframe) == 4
    assert dataframe.columns.tolist() == COMPACT_STUDY_COLUMNS
    assert list(zip(dataframe["alpha"], dataframe["beta"])) == [
        (1e-10, 0.0),
        (1e-10, 1e-7),
        (1e-9, 0.0),
        (1e-9, 1e-7),
    ]


def test_compact_parameter_study_is_scientifically_deterministic() -> None:
    first = run_small_compact_parameter_study()
    second = run_small_compact_parameter_study()

    pd.testing.assert_frame_equal(
        first.drop(columns="runtime_seconds"),
        second.drop(columns="runtime_seconds"),
        check_exact=True,
    )


def test_compact_parameter_study_builds_one_shared_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_forward = experiments_module.solve_forward
    original_reconstruct = (
        experiments_module.reconstruct_compact_nonnegative
    )
    forward_calls = 0
    sensor_data_ids: list[int] = []
    grid_ids: list[int] = []
    measurement_copies: list[np.ndarray] = []

    def tracking_forward(source, grid):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(source, grid)

    def tracking_reconstruct(
        sensor_data,
        grid,
        alpha,
        beta,
        *,
        max_iterations,
        tolerance,
    ):
        sensor_data_ids.append(id(sensor_data))
        grid_ids.append(id(grid))
        measurement_copies.append(sensor_data.values.copy())
        return original_reconstruct(
            sensor_data,
            grid,
            alpha=alpha,
            beta=beta,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    monkeypatch.setattr(
        experiments_module,
        "solve_forward",
        tracking_forward,
    )
    monkeypatch.setattr(
        experiments_module,
        "reconstruct_compact_nonnegative",
        tracking_reconstruct,
    )

    run_small_compact_parameter_study()

    assert forward_calls == 1
    assert len(set(sensor_data_ids)) == 1
    assert len(set(grid_ids)) == 1
    assert all(
        np.array_equal(values, measurement_copies[0])
        for values in measurement_copies[1:]
    )


def test_compact_parameter_study_metrics_are_finite() -> None:
    dataframe = run_small_compact_parameter_study()

    assert np.all(np.isfinite(dataframe.to_numpy(dtype=float)))
    assert np.all(dataframe["runtime_seconds"] >= 0.0)


def test_compact_parameter_study_compactness_is_consistent() -> None:
    dataframe = run_small_compact_parameter_study()
    interior_size = (8 - 2) * (8 - 2)

    assert np.all(
        dataframe["near_zero_count"]
        + dataframe["active_count"]
        == interior_size
    )
    assert np.allclose(
        dataframe["near_zero_fraction"],
        dataframe["near_zero_count"] / interior_size,
    )
    assert 0.0 in dataframe["beta"].to_numpy()


def test_compact_parameter_study_does_not_modify_sequences() -> None:
    alphas = [1e-10, 1e-9]
    betas = [0.0, 1e-7]
    original_alphas = alphas.copy()
    original_betas = betas.copy()

    run_compact_parameter_study(
        alphas,
        betas,
        grid_shape=(8, 8),
        num_sensors=9,
    )

    assert alphas == original_alphas
    assert betas == original_betas


@pytest.mark.parametrize(
    "invalid_alphas",
    [
        [],
        [0.0],
        [-1e-9],
        [float("nan")],
        [float("inf")],
        [True],
        [1e-9, 1e-9],
        "invalid",
    ],
)
def test_compact_parameter_study_rejects_invalid_alphas(
    invalid_alphas: object,
) -> None:
    with pytest.raises(ValidationError):
        run_compact_parameter_study(
            invalid_alphas,  # type: ignore[arg-type]
            [0.0],
        )


@pytest.mark.parametrize(
    "invalid_betas",
    [
        [],
        [-1e-7],
        [float("nan")],
        [float("inf")],
        [True],
        [0.0, 0.0],
        "invalid",
    ],
)
def test_compact_parameter_study_rejects_invalid_betas(
    invalid_betas: object,
) -> None:
    with pytest.raises(ValidationError):
        run_compact_parameter_study(
            [1e-9],
            invalid_betas,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_compact_parameter_study_rejects_invalid_threshold(
    invalid_threshold: object,
) -> None:
    with pytest.raises(ValidationError):
        run_compact_parameter_study(
            [1e-9],
            [0.0],
            near_zero_threshold=invalid_threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_max_iterations",
    [0, -1, 1.5, True],
)
def test_compact_parameter_study_rejects_invalid_iterations(
    invalid_max_iterations: object,
) -> None:
    with pytest.raises(ValidationError):
        run_compact_parameter_study(
            [1e-9],
            [0.0],
            max_iterations=invalid_max_iterations,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_tolerance",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_compact_parameter_study_rejects_invalid_tolerance(
    invalid_tolerance: object,
) -> None:
    with pytest.raises(ValidationError):
        run_compact_parameter_study(
            [1e-9],
            [0.0],
            tolerance=invalid_tolerance,  # type: ignore[arg-type]
        )


METHOD_STUDY_DETAILED_COLUMNS = [
    "sensor_count",
    "run_seed",
    "method",
    "relative_l2_error",
    "rmse",
    "mae",
    "max_absolute_error",
    "residual_norm",
    "relative_residual",
    "residual_rms",
    "solution_norm",
    "gradient_norm",
    "near_zero_count",
    "near_zero_fraction",
    "active_count",
    "source_min",
    "source_max",
    "runtime_seconds",
]

METHOD_STUDY_SUMMARY_COLUMNS = [
    "sensor_count",
    "method",
    "number_of_runs",
    "mean_relative_l2_error",
    "std_relative_l2_error",
    "mean_rmse",
    "std_rmse",
    "mean_residual_norm",
    "std_residual_norm",
    "mean_near_zero_fraction",
    "std_near_zero_fraction",
    "mean_runtime_seconds",
]

METHOD_ORDER = [
    "identity",
    "smooth_nonnegative",
    "compact_nonnegative",
]


@pytest.fixture(scope="module")
def small_method_study() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a small deterministic reconstruction-method study."""
    return run_reconstruction_method_study(
        seeds=[3, 7],
        sensor_counts=[4, 9],
        grid_shape=(8, 8),
        noise_level=0.01,
        seed=11,
    )


def test_method_study_columns_and_row_counts(
    small_method_study: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    detailed, summary = small_method_study

    assert detailed.columns.tolist() == METHOD_STUDY_DETAILED_COLUMNS
    assert summary.columns.tolist() == METHOD_STUDY_SUMMARY_COLUMNS
    assert len(detailed) == 2 * 2 * 3
    assert len(summary) == 2 * 3


def test_method_study_preserves_cartesian_and_method_order(
    small_method_study: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    detailed, summary = small_method_study
    expected = [
        (sensor_count, run_seed, method)
        for sensor_count in [4, 9]
        for run_seed in [3, 7]
        for method in METHOD_ORDER
    ]

    assert list(
        zip(
            detailed["sensor_count"],
            detailed["run_seed"],
            detailed["method"],
        )
    ) == expected
    assert list(zip(summary["sensor_count"], summary["method"])) == [
        (sensor_count, method)
        for sensor_count in [4, 9]
        for method in METHOD_ORDER
    ]


def test_method_study_uses_identical_data_across_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_identity = experiments_module.reconstruct_tikhonov
    original_smooth = experiments_module.reconstruct_smooth_tikhonov
    original_compact = (
        experiments_module.reconstruct_compact_nonnegative
    )
    sensor_ids: list[int] = []
    grid_ids: list[int] = []
    measurements: list[np.ndarray] = []

    def record(sensor_data, grid) -> None:
        sensor_ids.append(id(sensor_data))
        grid_ids.append(id(grid))
        measurements.append(sensor_data.values.copy())

    def tracking_identity(sensor_data, grid, alpha):
        record(sensor_data, grid)
        return original_identity(sensor_data, grid, alpha=alpha)

    def tracking_smooth(sensor_data, grid, alpha, *, nonnegative):
        record(sensor_data, grid)
        return original_smooth(
            sensor_data,
            grid,
            alpha=alpha,
            nonnegative=nonnegative,
        )

    def tracking_compact(
        sensor_data,
        grid,
        alpha,
        beta,
        *,
        max_iterations,
        tolerance,
    ):
        record(sensor_data, grid)
        return original_compact(
            sensor_data,
            grid,
            alpha=alpha,
            beta=beta,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    monkeypatch.setattr(
        experiments_module,
        "reconstruct_tikhonov",
        tracking_identity,
    )
    monkeypatch.setattr(
        experiments_module,
        "reconstruct_smooth_tikhonov",
        tracking_smooth,
    )
    monkeypatch.setattr(
        experiments_module,
        "reconstruct_compact_nonnegative",
        tracking_compact,
    )

    run_reconstruction_method_study(
        seeds=[3],
        sensor_counts=[4],
        grid_shape=(7, 7),
        noise_level=0.01,
        seed=11,
    )

    assert len(set(sensor_ids)) == 1
    assert len(set(grid_ids)) == 1
    assert all(
        np.array_equal(values, measurements[0])
        for values in measurements[1:]
    )


def test_method_study_is_scientifically_deterministic() -> None:
    settings = {
        "seeds": [5],
        "sensor_counts": [4],
        "grid_shape": (7, 7),
        "noise_level": 0.01,
        "seed": 13,
    }
    first_detailed, first_summary = (
        run_reconstruction_method_study(**settings)
    )
    second_detailed, second_summary = (
        run_reconstruction_method_study(**settings)
    )

    pd.testing.assert_frame_equal(
        first_detailed.drop(columns="runtime_seconds"),
        second_detailed.drop(columns="runtime_seconds"),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        first_summary.drop(columns="mean_runtime_seconds"),
        second_summary.drop(columns="mean_runtime_seconds"),
        check_exact=True,
    )


def test_method_study_metrics_and_compactness_are_consistent(
    small_method_study: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    detailed, summary = small_method_study
    interior_size = (8 - 2) * (8 - 2)
    detailed_numeric = detailed.drop(columns="method")
    summary_numeric = summary.drop(columns="method")

    assert np.all(
        np.isfinite(detailed_numeric.to_numpy(dtype=float))
    )
    assert np.all(
        np.isfinite(summary_numeric.to_numpy(dtype=float))
    )
    assert np.all(detailed["runtime_seconds"] >= 0.0)
    assert np.all(summary["mean_runtime_seconds"] >= 0.0)
    assert np.all(
        detailed["near_zero_count"] + detailed["active_count"]
        == interior_size
    )
    assert np.allclose(
        detailed["near_zero_fraction"],
        detailed["near_zero_count"] / interior_size,
    )

    nonnegative = detailed[
        detailed["method"].isin(
            ["smooth_nonnegative", "compact_nonnegative"]
        )
    ]
    assert np.all(nonnegative["source_min"] >= -1e-12)


def test_method_study_one_seed_has_zero_population_std() -> None:
    _, summary = run_reconstruction_method_study(
        seeds=[3],
        sensor_counts=[4],
        grid_shape=(7, 7),
    )
    std_columns = [
        "std_relative_l2_error",
        "std_rmse",
        "std_residual_norm",
        "std_near_zero_fraction",
    ]

    assert np.allclose(summary[std_columns], 0.0)


def test_method_study_does_not_modify_input_sequences() -> None:
    seeds = [3, 7]
    sensor_counts = [4, 9]
    original_seeds = seeds.copy()
    original_counts = sensor_counts.copy()

    run_reconstruction_method_study(
        seeds,
        sensor_counts,
        grid_shape=(8, 8),
    )

    assert seeds == original_seeds
    assert sensor_counts == original_counts


@pytest.mark.parametrize(
    "invalid_seeds",
    [[], [-1], [1.5], [True], [1, 1], "invalid"],
)
def test_method_study_rejects_invalid_seeds(
    invalid_seeds: object,
) -> None:
    with pytest.raises(ValidationError):
        run_reconstruction_method_study(
            invalid_seeds,  # type: ignore[arg-type]
            [4],
            grid_shape=(7, 7),
        )


@pytest.mark.parametrize(
    "invalid_counts",
    [[], [0], [-1], [1.5], [True], [4, 4], [26], "invalid"],
)
def test_method_study_rejects_invalid_sensor_counts(
    invalid_counts: object,
) -> None:
    with pytest.raises(ValidationError):
        run_reconstruction_method_study(
            [1],
            invalid_counts,  # type: ignore[arg-type]
            grid_shape=(7, 7),
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("identity_alpha", 0.0),
        ("identity_alpha", True),
        ("smooth_alpha", -1.0),
        ("smooth_alpha", float("nan")),
        ("compact_alpha", float("inf")),
        ("compact_alpha", 0.0),
        ("compact_beta", -1.0),
        ("compact_beta", True),
        ("compact_max_iterations", 0),
        ("compact_max_iterations", 1.5),
        ("compact_max_iterations", True),
        ("compact_tolerance", 0.0),
        ("compact_tolerance", float("nan")),
        ("near_zero_threshold", 0.0),
        ("near_zero_threshold", float("inf")),
    ],
)
def test_method_study_rejects_invalid_solver_parameters(
    parameter: str,
    value: object,
) -> None:
    settings = {
        "seeds": [1],
        "sensor_counts": [4],
        "grid_shape": (7, 7),
    }
    settings[parameter] = value

    with pytest.raises(ValidationError):
        run_reconstruction_method_study(
            **settings,  # type: ignore[arg-type]
        )


def test_synthetic_benchmark_defaults_to_identity_regularization() -> None:
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        seed=23,
    )

    assert result.config["regularization"] == "identity"


def test_explicit_identity_benchmark_matches_default() -> None:
    settings = {
        "grid_shape": (8, 8),
        "num_sensors": 9,
        "noise_level": 0.01,
        "alpha": 1e-4,
        "seed": 23,
    }

    default = run_synthetic_benchmark(**settings)
    explicit = run_synthetic_benchmark(
        **settings,
        regularization="identity",
    )

    assert np.array_equal(
        default.reconstructed_source,
        explicit.reconstructed_source,
    )
    assert np.array_equal(
        default.reconstruction.predicted_measurements,
        explicit.reconstruction.predicted_measurements,
    )
    assert explicit.config["regularization"] == "identity"


def test_synthetic_benchmark_supports_smoothness_regularization() -> None:
    result = run_synthetic_benchmark(
        grid_shape=(8, 9),
        num_sensors=10,
        noise_level=0.01,
        alpha=1e-4,
        regularization="smoothness",
        seed=23,
    )

    assert result.config["regularization"] == "smoothness"
    assert result.reconstructed_source.shape == (8, 9)
    assert np.all(np.isfinite(result.reconstructed_source))


def test_user_measurement_mode_supports_smoothness() -> None:
    grid, sensor_data = create_example_measurements()

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
        alpha=1e-4,
        regularization="smoothness",
    )

    assert result.config["regularization"] == "smoothness"
    assert np.all(np.isfinite(result.reconstructed_source))
    assert set(result.metrics) == {
        "residual_norm",
        "relative_residual",
        "residual_rms",
        "solution_norm",
    }
    assert "relative_l2_error" not in result.metrics


REGULARIZATION_COMPARISON_COLUMNS = [
    "regularization",
    "alpha",
    "sensor_count",
    "rmse",
    "mae",
    "relative_l2_error",
    "max_absolute_error",
    "residual_norm",
    "relative_residual",
    "residual_rms",
    "solution_norm",
    "runtime_seconds",
]

REGULARIZATION_COMPARISON_SETTINGS = {
    "alpha_by_method": {
        "smoothness": 1e-9,
        "identity": 1e-7,
    },
    "grid_shape": (10, 10),
    "num_sensors": 9,
    "noise_level": 0.01,
    "seed": 42,
}


@pytest.fixture(scope="module")
def small_regularization_comparison(
) -> tuple[pd.DataFrame, list[ExperimentResult]]:
    return run_regularization_comparison(
        **REGULARIZATION_COMPARISON_SETTINGS,
    )


def test_regularization_comparison_structure_and_configuration(
    small_regularization_comparison: tuple[
        pd.DataFrame,
        list[ExperimentResult],
    ],
) -> None:
    dataframe, results = small_regularization_comparison

    assert isinstance(dataframe, pd.DataFrame)
    assert isinstance(results, list)
    assert dataframe.columns.tolist() == REGULARIZATION_COMPARISON_COLUMNS
    assert len(dataframe) == 2
    assert len(results) == 2
    assert all(isinstance(result, ExperimentResult) for result in results)
    assert dataframe["regularization"].tolist() == [
        "identity",
        "smoothness",
    ]
    assert dataframe["alpha"].tolist() == [1e-7, 1e-9]

    for method, alpha, result in zip(
        ("identity", "smoothness"),
        (1e-7, 1e-9),
        results,
    ):
        assert result.config["mode"] == "synthetic_benchmark"
        assert result.config["study_type"] == (
            "regularization_comparison"
        )
        assert result.config["regularization"] == method
        assert result.config["alpha"] == alpha


def test_regularization_comparison_shares_benchmark_objects(
    small_regularization_comparison: tuple[
        pd.DataFrame,
        list[ExperimentResult],
    ],
) -> None:
    _, results = small_regularization_comparison
    identity, smoothness = results

    assert identity.grid is smoothness.grid
    assert identity.true_source is smoothness.true_source
    assert identity.temperature is smoothness.temperature
    assert identity.sensor_data_clean is smoothness.sensor_data_clean
    assert identity.sensor_data_noisy is smoothness.sensor_data_noisy
    assert np.array_equal(
        identity.sensor_data_clean.indices,
        smoothness.sensor_data_clean.indices,
    )
    assert np.array_equal(
        identity.sensor_data_noisy.indices,
        smoothness.sensor_data_noisy.indices,
    )
    assert np.array_equal(
        identity.sensor_data_noisy.values,
        smoothness.sensor_data_noisy.values,
    )


def test_regularization_comparison_generates_shared_data_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_forward = experiments_module.solve_forward
    original_noise = experiments_module.add_noise_to_sensor_data
    original_reconstruct = experiments_module.reconstruct_tikhonov
    forward_calls = 0
    noise_calls = 0
    reconstruction_calls: list[tuple[int, int, str, float]] = []

    def tracking_forward(source, grid):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(source, grid)

    def tracking_noise(sensor_data, *, noise_level, seed, relative):
        nonlocal noise_calls
        noise_calls += 1
        return original_noise(
            sensor_data,
            noise_level=noise_level,
            seed=seed,
            relative=relative,
        )

    def tracking_reconstruct(
        sensor_data,
        grid,
        alpha,
        *,
        regularization,
    ):
        reconstruction_calls.append(
            (id(sensor_data), id(grid), regularization, alpha)
        )
        return original_reconstruct(
            sensor_data,
            grid,
            alpha=alpha,
            regularization=regularization,
        )

    monkeypatch.setattr(
        experiments_module,
        "solve_forward",
        tracking_forward,
    )
    monkeypatch.setattr(
        experiments_module,
        "add_noise_to_sensor_data",
        tracking_noise,
    )
    monkeypatch.setattr(
        experiments_module,
        "reconstruct_tikhonov",
        tracking_reconstruct,
    )

    run_regularization_comparison(
        **REGULARIZATION_COMPARISON_SETTINGS,
    )

    assert forward_calls == 1
    assert noise_calls == 1
    assert len({call[0] for call in reconstruction_calls}) == 1
    assert len({call[1] for call in reconstruction_calls}) == 1
    assert [(call[2], call[3]) for call in reconstruction_calls] == [
        ("identity", 1e-7),
        ("smoothness", 1e-9),
    ]


def test_regularization_comparison_is_scientifically_deterministic() -> None:
    first, _ = run_regularization_comparison(
        **REGULARIZATION_COMPARISON_SETTINGS,
    )
    second, _ = run_regularization_comparison(
        **REGULARIZATION_COMPARISON_SETTINGS,
    )

    pd.testing.assert_frame_equal(
        first.drop(columns="runtime_seconds"),
        second.drop(columns="runtime_seconds"),
    )


def test_regularization_comparison_metrics_are_finite(
    small_regularization_comparison: tuple[
        pd.DataFrame,
        list[ExperimentResult],
    ],
) -> None:
    dataframe, _ = small_regularization_comparison
    metric_columns = REGULARIZATION_COMPARISON_COLUMNS[1:]

    assert np.all(np.isfinite(dataframe[metric_columns].to_numpy()))
    assert np.all(dataframe["runtime_seconds"] >= 0.0)


@pytest.mark.parametrize(
    ("alpha_by_method", "message"),
    [
        ({}, "not be empty"),
        ({"smoothness": 1e-9}, "identity"),
        ({"identity": 1e-7}, "smoothness"),
        (
            {"identity": 1e-7, "smoothness": 1e-9, "ridge": 1e-6},
            "unknown",
        ),
    ],
)
def test_regularization_comparison_rejects_invalid_mapping_keys(
    alpha_by_method: object,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        run_regularization_comparison(alpha_by_method)  # type: ignore[arg-type]


def test_regularization_comparison_rejects_non_mapping() -> None:
    with pytest.raises(ValidationError, match="mapping"):
        run_regularization_comparison(  # type: ignore[arg-type]
            [("identity", 1e-7), ("smoothness", 1e-9)]
        )


@pytest.mark.parametrize(
    "invalid_alpha",
    [0.0, -1e-7, True, np.nan, np.inf, -np.inf],
)
def test_regularization_comparison_rejects_invalid_alpha(
    invalid_alpha: object,
) -> None:
    with pytest.raises(ValidationError, match="alpha for identity"):
        run_regularization_comparison(
            {
                "identity": invalid_alpha,
                "smoothness": 1e-9,
            }
        )


def test_regularization_comparison_does_not_modify_mapping() -> None:
    alpha_by_method = {
        "smoothness": 1e-9,
        "identity": 1e-7,
    }
    original = alpha_by_method.copy()

    run_regularization_comparison(
        alpha_by_method,
        grid_shape=(8, 8),
        num_sensors=9,
        noise_level=0.01,
        seed=42,
    )

    assert alpha_by_method == original
    assert list(alpha_by_method) == list(original)


@pytest.fixture(scope="module")
def temperature_field_case() -> tuple[Grid2D, np.ndarray]:
    grid = Grid2D(nx=8, ny=9)
    source = gaussian_source(grid, sigma=0.12)
    return grid, solve_forward(source, grid)


@pytest.fixture(scope="module")
def default_temperature_field_result(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> MeasurementReconstructionResult:
    _, temperature = temperature_field_case
    return reconstruct_from_temperature_field(
        temperature,
        alpha=1e-4,
    )


def test_temperature_field_default_selection_and_configuration(
    temperature_field_case: tuple[Grid2D, np.ndarray],
    default_temperature_field_result: MeasurementReconstructionResult,
) -> None:
    grid, temperature = temperature_field_case
    result = default_temperature_field_result
    expected_indices = np.asarray(
        [
            (i, j)
            for i in range(1, grid.nx - 1)
            for j in range(1, grid.ny - 1)
        ]
    )
    expected_count = (grid.nx - 2) * (grid.ny - 2)

    assert isinstance(result, MeasurementReconstructionResult)
    assert result.reconstructed_source.shape == temperature.shape
    assert len(result.sensor_data) == expected_count
    assert result.reconstruction.n_sensors == expected_count
    assert np.array_equal(result.sensor_data.indices, expected_indices)
    assert np.array_equal(
        result.sensor_data.values,
        temperature[expected_indices[:, 0], expected_indices[:, 1]],
    )
    assert np.all(result.sensor_data.indices[:, 0] > 0)
    assert np.all(result.sensor_data.indices[:, 0] < grid.nx - 1)
    assert np.all(result.sensor_data.indices[:, 1] > 0)
    assert np.all(result.sensor_data.indices[:, 1] < grid.ny - 1)
    assert result.config == {
        "mode": "temperature_field",
        "grid_shape": temperature.shape,
        "domain_size": (1.0, 1.0),
        "num_sensors": expected_count,
        "num_measurements": expected_count,
        "measurement_selection": "all_interior",
        "alpha": 1e-4,
        "regularization": "identity",
    }


def test_temperature_field_custom_indices_preserve_order(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    _, temperature = temperature_field_case
    indices = [[6, 7], [1, 2], [4, 5]]

    result = reconstruct_from_temperature_field(
        temperature,
        sensor_indices=indices,
        alpha=2e-4,
    )

    assert np.array_equal(result.sensor_data.indices, indices)
    assert np.array_equal(
        result.sensor_data.values,
        temperature[[6, 1, 4], [7, 2, 5]],
    )
    assert result.reconstructed_source.shape == temperature.shape
    assert result.config["measurement_selection"] == "custom_indices"
    assert result.config["num_sensors"] == 3
    assert result.config["num_measurements"] == 3
    assert result.config["alpha"] == 2e-4
    assert result.config["regularization"] == "identity"


def test_temperature_field_supports_smoothness_regularization(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    _, temperature = temperature_field_case

    result = reconstruct_from_temperature_field(
        temperature,
        sensor_indices=[[1, 1], [3, 4], [6, 7]],
        alpha=1e-4,
        regularization="smoothness",
    )

    assert result.config["regularization"] == "smoothness"
    assert result.reconstructed_source.shape == temperature.shape
    assert np.all(np.isfinite(result.reconstructed_source))


def test_temperature_field_contains_measurement_metrics_only(
    default_temperature_field_result: MeasurementReconstructionResult,
) -> None:
    metrics = default_temperature_field_result.metrics

    assert set(metrics) == {
        "residual_norm",
        "relative_residual",
        "residual_rms",
        "solution_norm",
    }
    assert all(np.isfinite(value) for value in metrics.values())

    for source_metric in (
        "rmse",
        "mae",
        "relative_l2_error",
        "max_absolute_error",
    ):
        assert source_metric not in metrics
        assert source_metric not in default_temperature_field_result.config


def test_temperature_field_input_is_not_modified(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    _, temperature = temperature_field_case
    supplied_temperature = temperature.copy()
    original = supplied_temperature.copy()

    reconstruct_from_temperature_field(
        supplied_temperature,
        sensor_indices=[[2, 3]],
    )

    assert np.array_equal(supplied_temperature, original)


def test_temperature_field_reconstruction_is_deterministic(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    _, temperature = temperature_field_case
    settings = {
        "sensor_indices": [[1, 1], [4, 5]],
        "alpha": 1e-4,
    }

    first = reconstruct_from_temperature_field(temperature, **settings)
    second = reconstruct_from_temperature_field(temperature, **settings)

    assert np.array_equal(
        first.reconstructed_source,
        second.reconstructed_source,
    )
    assert np.array_equal(
        first.reconstruction.predicted_measurements,
        second.reconstruction.predicted_measurements,
    )
    assert first.metrics == second.metrics
    assert first.config == second.config


def test_temperature_field_accepts_matching_grid(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    grid, temperature = temperature_field_case

    result = reconstruct_from_temperature_field(
        temperature,
        grid=grid,
        sensor_indices=[[2, 3]],
    )

    assert result.grid.shape == grid.shape
    assert result.grid.domain.size == grid.domain.size


def test_temperature_field_accepts_custom_domain(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    _, temperature = temperature_field_case
    domain = Domain2D(length_x=2.0, length_y=3.0)

    result = reconstruct_from_temperature_field(
        temperature,
        domain=domain,
        sensor_indices=[[2, 3]],
    )

    assert result.grid.shape == temperature.shape
    assert result.grid.domain.size == domain.size
    assert result.config["domain_size"] == domain.size


@pytest.mark.parametrize(
    "invalid_field",
    [
        np.ones(5),
        np.ones((3, 3, 2)),
        np.empty((0, 3)),
        np.ones((2, 3)),
        np.ones((3, 2)),
        [["not-numeric"] * 3 for _ in range(3)],
        np.full((3, 3), np.nan),
        np.full((3, 3), np.inf),
        np.full((3, 3), -np.inf),
    ],
)
def test_temperature_field_rejects_invalid_field(
    invalid_field: object,
) -> None:
    with pytest.raises(ValidationError):
        reconstruct_from_temperature_field(invalid_field)  # type: ignore[arg-type]


def test_temperature_field_rejects_mismatching_grid_shape() -> None:
    with pytest.raises(ValidationError, match="shape"):
        reconstruct_from_temperature_field(
            np.ones((6, 7)),
            grid=Grid2D(nx=7, ny=6),
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("grid", object()),
        ("domain", object()),
    ],
)
def test_temperature_field_rejects_invalid_grid_or_domain(
    keyword: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        reconstruct_from_temperature_field(
            np.ones((5, 6)),
            **{keyword: value},
        )


def test_temperature_field_rejects_grid_and_domain_together() -> None:
    with pytest.raises(ValidationError, match="must not both"):
        reconstruct_from_temperature_field(
            np.ones((5, 6)),
            grid=Grid2D(nx=5, ny=6),
            domain=Domain2D(),
        )


@pytest.mark.parametrize(
    "invalid_indices",
    [
        [],
        [[1]],
        [[1, 2, 3]],
        [[-1, 1]],
        [[5, 1]],
        [[1, 1], [1, 1]],
        [[0, 1]],
        [[1, 5]],
        [[True, 1]],
        [[np.bool_(True), 1]],
        [[1.0, 1]],
        [["1", 1]],
        [[np.nan, 1]],
        [[np.inf, 1]],
        [[-np.inf, 1]],
    ],
)
def test_temperature_field_rejects_invalid_custom_indices(
    invalid_indices: object,
) -> None:
    with pytest.raises(ValidationError):
        reconstruct_from_temperature_field(
            np.ones((5, 6)),
            sensor_indices=invalid_indices,  # type: ignore[arg-type]
        )


def test_temperature_field_does_not_modify_custom_indices() -> None:
    indices = [[3, 4], [1, 2]]
    original = [pair.copy() for pair in indices]

    reconstruct_from_temperature_field(
        np.ones((5, 6)),
        sensor_indices=indices,
    )

    assert indices == original


def test_temperature_field_delegates_to_measurement_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = experiments_module.reconstruct_from_measurements
    calls: list[dict[str, object]] = []

    def tracking_reconstruction(
        sensor_data,
        *,
        grid_shape,
        domain,
        alpha,
        regularization,
    ):
        calls.append(
            {
                "sensor_data": sensor_data,
                "grid_shape": grid_shape,
                "domain": domain,
                "alpha": alpha,
                "regularization": regularization,
            }
        )
        return original(
            sensor_data,
            grid_shape=grid_shape,
            domain=domain,
            alpha=alpha,
            regularization=regularization,
        )

    monkeypatch.setattr(
        experiments_module,
        "reconstruct_from_measurements",
        tracking_reconstruction,
    )
    result = reconstruct_from_temperature_field(
        np.ones((5, 6)),
        sensor_indices=[[1, 2], [3, 4]],
        alpha=2e-4,
    )

    assert len(calls) == 1
    assert calls[0]["sensor_data"] is result.sensor_data
    assert calls[0]["grid_shape"] == (5, 6)
    assert isinstance(calls[0]["domain"], Domain2D)
    assert calls[0]["alpha"] == 2e-4
    assert calls[0]["regularization"] == "identity"


def test_measurement_workflow_accepts_correct_coordinates(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    grid, temperature = temperature_field_case
    sensor_data = create_sensor_data(
        temperature,
        np.array([[1, 2], [4, 5]]),
        grid,
    )

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
        domain=grid.domain,
        alpha=1e-4,
    )

    assert result.sensor_data is sensor_data
    assert result.reconstruction.n_sensors == 2
    assert result.config["regularization"] == "identity"


def test_phase_three_package_exports_are_complete_and_unique() -> None:
    import thermoreconlab
    from thermoreconlab import (
        ExperimentResult as PublicExperimentResult,
        MeasurementReconstructionResult as PublicMeasurementResult,
        reconstruct_compact_nonnegative,
        reconstruct_from_measurements as public_measurement_workflow,
        reconstruct_from_temperature_field as public_field_workflow,
        reconstruct_smooth_tikhonov,
        run_compact_parameter_study as public_compact_study,
        run_reconstruction_method_study as public_method_study,
        run_regularization_comparison as public_comparison,
        run_synthetic_benchmark as public_benchmark,
    )

    required_names = [
        "__version__",
        "ExperimentResult",
        "MeasurementReconstructionResult",
        "reconstruct_compact_nonnegative",
        "reconstruct_from_measurements",
        "reconstruct_from_temperature_field",
        "reconstruct_smooth_tikhonov",
        "run_compact_parameter_study",
        "run_reconstruction_method_study",
        "run_regularization_comparison",
        "run_synthetic_benchmark",
    ]
    imported_objects = [
        PublicExperimentResult,
        PublicMeasurementResult,
        reconstruct_compact_nonnegative,
        public_measurement_workflow,
        public_field_workflow,
        reconstruct_smooth_tikhonov,
        public_compact_study,
        public_method_study,
        public_comparison,
        public_benchmark,
    ]

    assert thermoreconlab.__version__ == "0.1.0"
    assert all(imported_objects)
    assert len(thermoreconlab.__all__) == len(set(thermoreconlab.__all__))
    assert all(
        thermoreconlab.__all__.count(name) == 1
        for name in required_names
    )

    for private_name in (
        "_validate_regularization_alphas",
        "_validate_temperature_field",
        "_validate_temperature_sensor_indices",
        "_build_interior_gradient_matrix",
    ):
        assert private_name not in thermoreconlab.__all__


def test_measurement_workflow_accepts_coordinates_within_tolerance(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    grid, temperature = temperature_field_case
    indices = np.array([[1, 2], [4, 5]])
    exact = create_sensor_data(temperature, indices, grid)
    sensor_data = SensorData(
        indices=exact.indices,
        values=exact.values,
        coordinates=exact.coordinates + 0.5e-6,
    )

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
        domain=grid.domain,
    )

    assert result.reconstruction.n_sensors == 2


@pytest.mark.parametrize(
    ("axis", "column"),
    [("x", 0), ("y", 1)],
)
def test_measurement_workflow_rejects_incorrect_coordinates(
    axis: str,
    column: int,
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    grid, temperature = temperature_field_case
    exact = create_sensor_data(
        temperature,
        np.array([[1, 2], [4, 5]]),
        grid,
    )
    coordinates = exact.coordinates.copy()
    coordinates[1, column] += 0.01
    sensor_data = SensorData(
        indices=exact.indices,
        values=exact.values,
        coordinates=coordinates,
    )

    with pytest.raises(ValidationError, match=rf"{axis} coordinate"):
        reconstruct_from_measurements(
            sensor_data,
            grid_shape=grid.shape,
            domain=grid.domain,
        )


def test_measurement_workflow_accepts_sensor_data_without_coordinates(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    grid, temperature = temperature_field_case
    indices = np.array([[1, 2], [4, 5]])
    sensor_data = SensorData(
        indices=indices,
        values=temperature[indices[:, 0], indices[:, 1]],
    )

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
        domain=grid.domain,
    )

    assert result.sensor_data.coordinates is None
    assert result.reconstruction.n_sensors == 2


def test_measurement_workflow_rejects_all_boundary_sensors(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    grid, temperature = temperature_field_case
    indices = np.array([[0, 2], [grid.nx - 1, 4], [3, 0]])
    sensor_data = create_sensor_data(temperature, indices, grid)

    with pytest.raises(
        ValidationError,
        match="no source-identifying information",
    ):
        reconstruct_from_measurements(
            sensor_data,
            grid_shape=grid.shape,
            domain=grid.domain,
        )


def test_measurement_workflow_preserves_mixed_boundary_sensors(
    temperature_field_case: tuple[Grid2D, np.ndarray],
) -> None:
    grid, temperature = temperature_field_case
    indices = np.array([[0, 2], [3, 4], [grid.nx - 1, 6]])
    sensor_data = create_sensor_data(temperature, indices, grid)

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
        domain=grid.domain,
        alpha=1e-4,
    )

    assert result.sensor_data is sensor_data
    assert np.array_equal(result.sensor_data.indices, indices)
    assert result.reconstruction.n_sensors == len(indices)
    assert result.reconstruction.predicted_measurements.shape == (3,)
    assert set(result.metrics) == {
        "residual_norm",
        "relative_residual",
        "residual_rms",
        "solution_norm",
    }
    for source_metric in (
        "rmse",
        "mae",
        "relative_l2_error",
        "max_absolute_error",
    ):
        assert source_metric not in result.metrics


@pytest.mark.parametrize("regularization", ["identity", "smoothness"])
def test_measurement_workflow_regularizers_remain_compatible(
    regularization: str,
) -> None:
    grid = Grid2D(nx=5, ny=6)
    source = gaussian_source(grid, sigma=0.12)
    temperature = solve_forward(source, grid)
    sensor_data = create_sensor_data(
        temperature,
        np.array([[1, 2], [3, 4]]),
        grid,
    )

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
        domain=grid.domain,
        alpha=1e-4,
        regularization=regularization,
    )

    assert result.config["regularization"] == regularization
    assert result.reconstructed_source.shape == grid.shape


def test_temperature_field_measurements_remain_interior_only(
    default_temperature_field_result: MeasurementReconstructionResult,
) -> None:
    result = default_temperature_field_result
    indices = result.sensor_data.indices

    assert result.config["mode"] == "temperature_field"
    assert np.all(indices[:, 0] > 0)
    assert np.all(indices[:, 0] < result.grid.nx - 1)
    assert np.all(indices[:, 1] > 0)
    assert np.all(indices[:, 1] < result.grid.ny - 1)


def test_csv_sensor_coordinates_reconstruct_successfully(tmp_path) -> None:
    grid = Grid2D(nx=5, ny=6)
    source = gaussian_source(grid, sigma=0.12)
    temperature = solve_forward(source, grid)
    indices = np.array([[1, 2], [3, 4]])
    path = tmp_path / "sensors_with_coordinates.csv"
    pd.DataFrame(
        {
            "i": indices[:, 0],
            "j": indices[:, 1],
            "value": temperature[indices[:, 0], indices[:, 1]],
            "x": grid.x[indices[:, 0]],
            "y": grid.y[indices[:, 1]],
        }
    ).to_csv(path, index=False)
    sensor_data = load_sensor_csv(path)

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
        domain=grid.domain,
    )

    assert result.sensor_data is sensor_data
    assert result.reconstruction.n_sensors == 2
