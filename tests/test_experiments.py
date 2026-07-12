"""Tests for the high-level experiment workflow."""

import numpy as np
import pytest

from thermoreconlab import run_synthetic_benchmark
from thermoreconlab.experiments import ExperimentResult
from thermoreconlab.exceptions import ValidationError


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
