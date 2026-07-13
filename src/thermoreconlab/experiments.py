"""High-level experiment workflows for ThermoReconLab.

This module connects the numerical core, synthetic data generation,
sensor handling, inverse reconstruction, and validation metrics into
simple user-facing experiments.
"""

from __future__ import annotations
from collections.abc import Sequence

import pandas as pd
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from thermoreconlab.analysis import compute_all_metrics
from thermoreconlab.core.domain import Domain2D
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.data import (
    gaussian_source,
    random_hotspots,
    two_gaussian_sources,
)
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.reconstruction import (
    ReconstructionResult,
    reconstruct_tikhonov,
    solve_forward,
)
from thermoreconlab.sensors import (
    SensorData,
    add_noise_to_sensor_data,
    center_focused_sensors,
    create_sensor_data,
    random_sensors,
    regular_grid_sensors,
)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Store the complete result of a synthetic benchmark."""

    grid: Grid2D
    true_source: NDArray[np.float64]
    temperature: NDArray[np.float64]
    sensor_data_clean: SensorData
    sensor_data_noisy: SensorData
    reconstruction: ReconstructionResult
    metrics: dict[str, float]
    config: dict[str, Any]
    runtime: float

    @property
    def reconstructed_source(self) -> NDArray[np.float64]:
        """Return the reconstructed heat-source field."""
        return self.reconstruction.source

    def to_dict(self) -> dict[str, Any]:
        """Return a compact serializable experiment summary."""
        return {
            "config": dict(self.config),
            "metrics": dict(self.metrics),
            "runtime": float(self.runtime),
            "reconstruction": {
                "alpha": float(self.reconstruction.alpha),
                "residual_norm": float(
                    self.reconstruction.residual_norm
                ),
                "solution_norm": float(
                    self.reconstruction.solution_norm
                ),
                "runtime": float(self.reconstruction.runtime),
                "n_sensors": int(self.reconstruction.n_sensors),
            },
        }

@dataclass(frozen=True, slots=True)
class MeasurementReconstructionResult:
    """Store a reconstruction produced from user measurements.

    Unlike a synthetic benchmark result, this object does not contain
    a true source or ground-truth error metrics.
    """

    grid: Grid2D
    sensor_data: SensorData
    reconstruction: ReconstructionResult
    config: dict[str, Any]
    runtime: float

    @property
    def reconstructed_source(self) -> NDArray[np.float64]:
        """Return the reconstructed heat-source field."""
        return self.reconstruction.source

    def to_dict(self) -> dict[str, Any]:
        """Return a compact serializable reconstruction summary."""
        return {
            "config": dict(self.config),
            "runtime": float(self.runtime),
            "reconstruction": {
                "alpha": float(self.reconstruction.alpha),
                "residual_norm": float(
                    self.reconstruction.residual_norm
                ),
                "solution_norm": float(
                    self.reconstruction.solution_norm
                ),
                "runtime": float(self.reconstruction.runtime),
                "n_sensors": int(self.reconstruction.n_sensors),
            },
        }
    
def _validate_grid_shape(
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    """Validate and normalize a two-dimensional grid shape."""
    if (
        not isinstance(grid_shape, tuple)
        or len(grid_shape) != 2
    ):
        raise ValidationError(
            "grid_shape must be a tuple containing (nx, ny)."
        )

    normalized: list[int] = []

    for name, value in zip(("nx", "ny"), grid_shape):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValidationError(f"{name} must be an integer.")

        integer_value = int(value)

        if integer_value < 3:
            raise ValidationError(f"{name} must be at least 3.")

        normalized.append(integer_value)

    return normalized[0], normalized[1]


def _validate_nonnegative_real(
    value: Real,
    name: str,
) -> float:
    """Validate a finite nonnegative real parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number.")

    result = float(value)

    if not np.isfinite(result):
        raise ValidationError(f"{name} must be finite.")

    if result < 0.0:
        raise ValidationError(f"{name} must be nonnegative.")

    return result


def _normalize_choice(value: str, name: str) -> str:
    """Normalize a user-provided strategy name."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string.")

    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _derived_seeds(
    seed: int | None,
) -> tuple[int, int, int]:
    """Create independent deterministic seeds for one experiment."""
    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "seed must be an integer or None."
        ) from error

    maximum = np.iinfo(np.uint32).max

    return tuple(
        int(value)
        for value in rng.integers(
            0,
            maximum,
            size=3,
            dtype=np.uint32,
        )
    )


def _create_synthetic_source(
    grid: Grid2D,
    source_type: str,
    *,
    seed: int,
) -> NDArray[np.float64]:
    """Generate one of the supported benchmark source fields."""
    source_name = _normalize_choice(source_type, "source_type")
    length_scale = min(
        grid.domain.length_x,
        grid.domain.length_y,
    )

    if source_name in {"gaussian", "single_gaussian"}:
        return gaussian_source(
            grid,
            center=(
                0.50 * grid.domain.length_x,
                0.50 * grid.domain.length_y,
            ),
            amplitude=1.0,
            sigma=0.08 * length_scale,
        )

    if source_name in {"two_gaussians", "double_gaussian"}:
        return two_gaussian_sources(
            grid,
            centers=(
                (
                    0.35 * grid.domain.length_x,
                    0.40 * grid.domain.length_y,
                ),
                (
                    0.70 * grid.domain.length_x,
                    0.65 * grid.domain.length_y,
                ),
            ),
            amplitudes=(1.0, 0.7),
            sigmas=(
                0.07 * length_scale,
                0.09 * length_scale,
            ),
        )

    if source_name == "random_hotspots":
        return random_hotspots(
            grid,
            n_hotspots=3,
            seed=seed,
            sigma_range=(
                0.04 * length_scale,
                0.10 * length_scale,
            ),
        )

    raise ValidationError(
        "Unsupported source_type. Choose 'gaussian', "
        "'two_gaussians', or 'random_hotspots'."
    )


def _place_sensors(
    grid: Grid2D,
    sensor_strategy: str,
    num_sensors: int,
    *,
    seed: int,
) -> NDArray[np.int64]:
    """Place sensors using one of the supported useful strategies."""
    strategy = _normalize_choice(
        sensor_strategy,
        "sensor_strategy",
    )

    if strategy == "regular":
        return regular_grid_sensors(
            grid,
            count=num_sensors,
            include_boundary=False,
        )

    if strategy == "random":
        return random_sensors(
            grid,
            count=num_sensors,
            seed=seed,
            include_boundary=False,
        )

    if strategy in {"center", "center_focused"}:
        return center_focused_sensors(
            grid,
            count=num_sensors,
            seed=seed,
            include_boundary=False,
        )

    if strategy == "boundary":
        raise ValidationError(
            "Boundary-only sensors are not supported by the standard "
            "benchmark because homogeneous Dirichlet boundary "
            "temperatures are fixed at zero and contain no source "
            "information."
        )

    raise ValidationError(
        "Unsupported sensor_strategy. Choose 'regular', 'random', "
        "or 'center_focused'."
    )


def run_synthetic_benchmark(
    *,
    grid_shape: tuple[int, int] = (30, 30),
    domain: Domain2D | None = None,
    source_type: str = "two_gaussians",
    sensor_strategy: str = "regular",
    num_sensors: int = 25,
    noise_level: Real = 0.02,
    alpha: Real = 1e-3,
    seed: int | None = 42,
) -> ExperimentResult:
    """Run a complete reproducible synthetic reconstruction benchmark."""
    nx, ny = _validate_grid_shape(grid_shape)
    noise_value = _validate_nonnegative_real(
        noise_level,
        "noise_level",
    )

    if domain is None:
        selected_domain = Domain2D()
    elif isinstance(domain, Domain2D):
        selected_domain = domain
    else:
        raise ValidationError(
            "domain must be a Domain2D object or None."
        )

    source_seed, sensor_seed, noise_seed = _derived_seeds(seed)

    start_time = perf_counter()

    grid = Grid2D(
        nx=nx,
        ny=ny,
        domain=selected_domain,
    )

    true_source = _create_synthetic_source(
        grid,
        source_type,
        seed=source_seed,
    )

    temperature = solve_forward(true_source, grid)

    sensor_indices = _place_sensors(
        grid,
        sensor_strategy,
        num_sensors,
        seed=sensor_seed,
    )

    sensor_data_clean = create_sensor_data(
        temperature,
        sensor_indices,
        grid,
    )

    sensor_data_noisy = add_noise_to_sensor_data(
        sensor_data_clean,
        noise_level=noise_value,
        seed=noise_seed,
        relative=True,
    )

    reconstruction = reconstruct_tikhonov(
        sensor_data_noisy,
        grid,
        alpha=alpha,
    )

    metrics = compute_all_metrics(
        true_source,
        reconstruction.source,
    )
    metrics["residual_norm"] = float(
        reconstruction.residual_norm
    )
    metrics["solution_norm"] = float(
        reconstruction.solution_norm
    )

    runtime = perf_counter() - start_time




    config: dict[str, Any] = {
    "mode": "synthetic_benchmark",
    "grid_shape": grid.shape,
    "domain_size": grid.domain.size,
    "source_type": _normalize_choice(
        source_type,
        "source_type",
    ),
    "sensor_strategy": _normalize_choice(
        sensor_strategy,
        "sensor_strategy",
    ),
    "num_sensors": int(num_sensors),
    "noise_level": noise_value,
    "alpha": float(reconstruction.alpha),
    "seed": seed,
}


    return ExperimentResult(
        grid=grid,
        true_source=true_source,
        temperature=temperature,
        sensor_data_clean=sensor_data_clean,
        sensor_data_noisy=sensor_data_noisy,
        reconstruction=reconstruction,
        metrics=metrics,
        config=config,
        runtime=float(runtime),
    )
def reconstruct_from_measurements(
    sensor_data: SensorData,
    *,
    grid_shape: tuple[int, int] = (30, 30),
    domain: Domain2D | None = None,
    alpha: Real = 1e-3,
) -> MeasurementReconstructionResult:
    """Reconstruct a heat source from user-provided measurements.

    The supplied sensor indices are interpreted on the requested grid.
    No synthetic source is generated and no ground-truth source-error
    metrics are calculated.

    Parameters
    ----------
    sensor_data:
        User-provided grid indices and measured temperatures.
    grid_shape:
        Number of grid points as ``(nx, ny)``.
    domain:
        Optional physical domain. The unit square is used by default.
    alpha:
        Positive Tikhonov regularization parameter.

    Returns
    -------
    MeasurementReconstructionResult
        Reconstructed source, grid, measurements, and diagnostics.
    """
    if not isinstance(sensor_data, SensorData):
        raise ValidationError(
            "sensor_data must be a SensorData object."
        )

    nx, ny = _validate_grid_shape(grid_shape)

    if domain is None:
        selected_domain = Domain2D()
    elif isinstance(domain, Domain2D):
        selected_domain = domain
    else:
        raise ValidationError(
            "domain must be a Domain2D object or None."
        )

    start_time = perf_counter()

    grid = Grid2D(
        nx=nx,
        ny=ny,
        domain=selected_domain,
    )

    reconstruction = reconstruct_tikhonov(
        sensor_data,
        grid,
        alpha=alpha,
    )

    runtime = perf_counter() - start_time

    config: dict[str, Any] = {
        "mode": "user_measurements",
        "grid_shape": grid.shape,
        "domain_size": grid.domain.size,
        "num_sensors": len(sensor_data),
        "alpha": float(reconstruction.alpha),
    }

    return MeasurementReconstructionResult(
        grid=grid,
        sensor_data=sensor_data,
        reconstruction=reconstruction,
        config=config,
        runtime=float(runtime),
    )
def run_regularization_study(
        
    alpha_values: Sequence[Real],
    *,
    grid_shape: tuple[int, int] = (30, 30),
    domain: Domain2D | None = None,
    source_type: str = "two_gaussians",
    sensor_strategy: str = "regular",
    num_sensors: int = 25,
    noise_level: Real = 0.02,
    seed: int | None = 42,
) -> tuple[pd.DataFrame, list[ExperimentResult]]:
    """Compare several Tikhonov regularization parameters.

    The same source, sensor locations, and noisy measurements are used
    for every alpha value so that the comparison remains fair.

    Parameters
    ----------
    alpha_values:
        Positive regularization parameters to evaluate.
    grid_shape:
        Number of grid points as ``(nx, ny)``.
    domain:
        Optional physical domain.
    source_type:
        Synthetic benchmark source type.
    sensor_strategy:
        Sensor placement strategy.
    num_sensors:
        Number of sparse temperature sensors.
    noise_level:
        Relative Gaussian measurement noise.
    seed:
        Master random seed.

    Returns
    -------
    tuple[pandas.DataFrame, list[ExperimentResult]]
        Study table and complete experiment results.
    """
    if isinstance(alpha_values, (str, bytes)):
        raise ValidationError(
            "alpha_values must be a sequence of numbers."
        )

    try:
        alpha_list = list(alpha_values)
    except TypeError as error:
        raise ValidationError(
            "alpha_values must be a sequence of numbers."
        ) from error

    if not alpha_list:
        raise ValidationError(
            "alpha_values must contain at least one value."
        )

    results: list[ExperimentResult] = []
    rows: list[dict[str, Any]] = []

    for alpha in alpha_list:
        result = run_synthetic_benchmark(
            grid_shape=grid_shape,
            domain=domain,
            source_type=source_type,
            sensor_strategy=sensor_strategy,
            num_sensors=num_sensors,
            noise_level=noise_level,
            alpha=alpha,
            seed=seed,
        )

        results.append(result)

        rows.append(
            {
                "study_type": "regularization",
                "alpha": result.reconstruction.alpha,
                "rmse": result.metrics["rmse"],
                "mae": result.metrics["mae"],
                "relative_l2_error": result.metrics[
                    "relative_l2_error"
                ],
                "max_absolute_error": result.metrics[
                    "max_absolute_error"
                ],
                "residual_norm": result.reconstruction.residual_norm,
                "solution_norm": result.reconstruction.solution_norm,
                "max_reconstructed_source": float(
                    np.max(result.reconstructed_source)
                ),
                "runtime": result.runtime,
            }
        )

    dataframe = pd.DataFrame(rows)

    return dataframe, results


def run_sensor_count_study(
    sensor_counts: Sequence[int],
    *,
    grid_shape: tuple[int, int] = (30, 30),
    domain: Domain2D | None = None,
    source_type: str = "two_gaussians",
    sensor_strategy: str = "regular",
    noise_level: Real = 0.02,
    alpha: Real = 1e-7,
    seed: int | None = 42,
) -> tuple[pd.DataFrame, list[ExperimentResult]]:
    """Compare reconstruction quality for different sensor counts.

    The same source configuration, noise level, regularization
    parameter, and random seed are used for every run.

    Parameters
    ----------
    sensor_counts:
        Positive numbers of sensors to evaluate.
    grid_shape:
        Number of grid points as ``(nx, ny)``.
    domain:
        Optional physical domain.
    source_type:
        Synthetic source configuration.
    sensor_strategy:
        Sensor placement strategy.
    noise_level:
        Relative Gaussian measurement noise.
    alpha:
        Tikhonov regularization parameter.
    seed:
        Master random seed.

    Returns
    -------
    tuple[pandas.DataFrame, list[ExperimentResult]]
        Study table and complete benchmark results.
    """
    if isinstance(sensor_counts, (str, bytes)):
        raise ValidationError(
            "sensor_counts must be a sequence of positive integers."
        )

    try:
        count_list = list(sensor_counts)
    except TypeError as error:
        raise ValidationError(
            "sensor_counts must be a sequence of positive integers."
        ) from error

    if not count_list:
        raise ValidationError(
            "sensor_counts must contain at least one value."
        )

    validated_counts: list[int] = []

    for count in count_list:
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or int(count) <= 0
        ):
            raise ValidationError(
                "Every sensor count must be a positive integer."
            )

        validated_counts.append(int(count))

    results: list[ExperimentResult] = []
    rows: list[dict[str, Any]] = []

    for sensor_count in validated_counts:
        result = run_synthetic_benchmark(
            grid_shape=grid_shape,
            domain=domain,
            source_type=source_type,
            sensor_strategy=sensor_strategy,
            num_sensors=sensor_count,
            noise_level=noise_level,
            alpha=alpha,
            seed=seed,
        )

        results.append(result)

        actual_count = result.reconstruction.n_sensors

        rows.append(
            {
                "study_type": "sensor_count",
                "sensor_count": actual_count,
                "sensor_fraction": (
                    actual_count / result.grid.size
                ),
                "rmse": result.metrics["rmse"],
                "mae": result.metrics["mae"],
                "relative_l2_error": result.metrics[
                    "relative_l2_error"
                ],
                "max_absolute_error": result.metrics[
                    "max_absolute_error"
                ],
                "residual_norm": result.reconstruction.residual_norm,
                "solution_norm": result.reconstruction.solution_norm,
                "runtime": result.runtime,
            }
        )

    dataframe = pd.DataFrame(rows)

    return dataframe, results


def run_noise_sensitivity_study(
    noise_levels: Sequence[Real],
    *,
    grid_shape: tuple[int, int] = (30, 30),
    domain: Domain2D | None = None,
    source_type: str = "two_gaussians",
    sensor_strategy: str = "regular",
    num_sensors: int = 25,
    alpha: Real = 1e-7,
    seed: int | None = 42,
) -> tuple[pd.DataFrame, list[ExperimentResult]]:
    """Compare reconstruction quality at different noise levels.

    The same true source, sensor locations, and random-noise pattern
    are used for every run. Only the noise magnitude changes.

    Parameters
    ----------
    noise_levels:
        Nonnegative relative Gaussian noise levels.
    grid_shape:
        Number of grid points as ``(nx, ny)``.
    domain:
        Optional physical domain.
    source_type:
        Synthetic benchmark source type.
    sensor_strategy:
        Sensor placement strategy.
    num_sensors:
        Number of sparse temperature sensors.
    alpha:
        Tikhonov regularization parameter.
    seed:
        Master random seed.

    Returns
    -------
    tuple[pandas.DataFrame, list[ExperimentResult]]
        Study table and complete benchmark results.
    """
    if isinstance(noise_levels, (str, bytes)):
        raise ValidationError(
            "noise_levels must be a sequence of nonnegative numbers."
        )

    try:
        level_list = list(noise_levels)
    except TypeError as error:
        raise ValidationError(
            "noise_levels must be a sequence of nonnegative numbers."
        ) from error

    if not level_list:
        raise ValidationError(
            "noise_levels must contain at least one value."
        )

    validated_levels = [
        _validate_nonnegative_real(level, "noise level")
        for level in level_list
    ]

    results: list[ExperimentResult] = []
    rows: list[dict[str, Any]] = []

    for noise_level in validated_levels:
        result = run_synthetic_benchmark(
            grid_shape=grid_shape,
            domain=domain,
            source_type=source_type,
            sensor_strategy=sensor_strategy,
            num_sensors=num_sensors,
            noise_level=noise_level,
            alpha=alpha,
            seed=seed,
        )

        measurement_difference = (
            result.sensor_data_noisy.values
            - result.sensor_data_clean.values
        )

        results.append(result)

        rows.append(
            {
                "study_type": "noise_sensitivity",
                "noise_level": noise_level,
                "measurement_noise_norm": float(
                    np.linalg.norm(measurement_difference)
                ),
                "mean_absolute_measurement_noise": float(
                    np.mean(np.abs(measurement_difference))
                ),
                "rmse": result.metrics["rmse"],
                "mae": result.metrics["mae"],
                "relative_l2_error": result.metrics[
                    "relative_l2_error"
                ],
                "max_absolute_error": result.metrics[
                    "max_absolute_error"
                ],
                "residual_norm": result.reconstruction.residual_norm,
                "solution_norm": result.reconstruction.solution_norm,
                "runtime": result.runtime,
            }
        )

    dataframe = pd.DataFrame(rows)

    return dataframe, results
