"""High-level experiment workflows for ThermoReconLab.

This module connects the numerical core, synthetic data generation,
sensor handling, inverse reconstruction, and validation metrics into
simple user-facing experiments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from thermoreconlab.analysis import (
    compute_all_metrics,
    relative_residual,
    residual_rms,
)
from thermoreconlab.core.domain import Domain2D
from thermoreconlab.core.fields import ensure_2d_array
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.data import (
    gaussian_source,
    random_hotspots,
    two_gaussian_sources,
)
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.reconstruction import (
    ReconstructionResult,
    reconstruct_compact_nonnegative,
    reconstruct_smooth_tikhonov,
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
    validate_sensor_data_for_grid,
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

    @property
    def metrics(self) -> dict[str, float]:
        """Return measurement-space reconstruction diagnostics."""
        return _measurement_metrics(
            self.reconstruction,
            self.sensor_data.values,
        )

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


def _measurement_metrics(
    reconstruction: ReconstructionResult,
    observed_measurements: NDArray[np.float64],
) -> dict[str, float]:
    """Collect measurement-space reconstruction diagnostics."""
    predicted = reconstruction.predicted_measurements

    return {
        "residual_norm": float(reconstruction.residual_norm),
        "relative_residual": relative_residual(
            predicted,
            observed_measurements,
        ),
        "residual_rms": residual_rms(
            predicted,
            observed_measurements,
        ),
        "solution_norm": float(reconstruction.solution_norm),
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


def _validate_interior_sensor_indices(
    sensor_indices: Sequence[Sequence[Integral]],
    grid: Grid2D,
) -> NDArray[np.int64]:
    """Validate and copy ordered interior grid-index pairs."""
    if isinstance(sensor_indices, (str, bytes)):
        raise ValidationError(
            "sensor_indices must be a nonempty sequence of index pairs."
        )

    try:
        entries = list(sensor_indices)
    except TypeError as error:
        raise ValidationError(
            "sensor_indices must be a nonempty sequence of index pairs."
        ) from error

    if not entries:
        raise ValidationError(
            "sensor_indices must contain at least one index pair."
        )

    validated: list[tuple[int, int]] = []

    for entry in entries:
        if isinstance(entry, (str, bytes)):
            raise ValidationError(
                "Each sensor index must contain exactly two integers."
            )

        try:
            pair = list(entry)
        except TypeError as error:
            raise ValidationError(
                "Each sensor index must contain exactly two integers."
            ) from error

        if len(pair) != 2:
            raise ValidationError(
                "Each sensor index must contain exactly two integers."
            )

        clean_pair: list[int] = []

        for value in pair:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                Integral,
            ):
                raise ValidationError(
                    "Sensor indices must contain integer values; "
                    "booleans and floating-point values are invalid."
                )

            clean_pair.append(int(value))

        i, j = clean_pair

        if i < 0 or j < 0:
            raise ValidationError(
                "Sensor indices must be nonnegative."
            )

        if i >= grid.nx or j >= grid.ny:
            raise ValidationError(
                "Sensor indices must lie within the grid."
            )

        if i in {0, grid.nx - 1} or j in {0, grid.ny - 1}:
            raise ValidationError(
                "Temperature-field sensor indices must be interior nodes."
            )

        validated.append((i, j))

    if len(set(validated)) != len(validated):
        raise ValidationError(
            "sensor_indices must not contain duplicate pairs."
        )

    return np.asarray(validated, dtype=np.int64)


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


def _validate_seed_values(seeds: Sequence[int]) -> list[int]:
    """Validate a nonempty sequence of nonnegative integer seeds."""
    if isinstance(seeds, (str, bytes)):
        raise ValidationError(
            "seeds must be a sequence of nonnegative integers."
        )

    try:
        seed_list = list(seeds)
    except TypeError as error:
        raise ValidationError(
            "seeds must be a sequence of nonnegative integers."
        ) from error

    if not seed_list:
        raise ValidationError(
            "seeds must contain at least one value."
        )

    validated_seeds: list[int] = []

    for seed in seed_list:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValidationError(
                "Every seed must be a nonnegative integer."
            )

        integer_seed = int(seed)

        if integer_seed < 0:
            raise ValidationError(
                "Every seed must be a nonnegative integer."
            )

        validated_seeds.append(integer_seed)

    return validated_seeds


def _validate_positive_real(
    value: Real,
    name: str,
) -> float:
    """Validate a finite positive real parameter."""
    result = _validate_nonnegative_real(value, name)

    if result == 0.0:
        raise ValidationError(f"{name} must be greater than zero.")

    return result


def _validate_regularization_alphas(
    alpha_by_method: Mapping[str, Real],
) -> dict[Literal["identity", "smoothness"], float]:
    """Validate and copy method-specific comparison parameters."""
    if not isinstance(alpha_by_method, Mapping):
        raise ValidationError("alpha_by_method must be a mapping.")

    if not alpha_by_method:
        raise ValidationError("alpha_by_method must not be empty.")

    required_methods = ("identity", "smoothness")
    unknown_methods = [
        method
        for method in alpha_by_method
        if method not in required_methods
    ]

    if unknown_methods:
        raise ValidationError(
            "alpha_by_method contains unknown method(s): "
            + ", ".join(repr(method) for method in unknown_methods)
            + "."
        )

    missing_methods = [
        method
        for method in required_methods
        if method not in alpha_by_method
    ]

    if missing_methods:
        raise ValidationError(
            "alpha_by_method is missing method(s): "
            + ", ".join(missing_methods)
            + "."
        )

    return {
        method: _validate_positive_real(
            alpha_by_method[method],
            f"alpha for {method}",
        )
        for method in required_methods
    }


def _validate_compact_parameter_values(
    values: Sequence[Real],
    name: str,
    *,
    positive: bool,
) -> list[float]:
    """Validate and copy one compact-study parameter sequence."""
    if isinstance(values, (str, bytes)):
        raise ValidationError(
            f"{name} must be a sequence of real numbers."
        )

    try:
        value_list = list(values)
    except TypeError as error:
        raise ValidationError(
            f"{name} must be a sequence of real numbers."
        ) from error

    if not value_list:
        raise ValidationError(
            f"{name} must contain at least one value."
        )

    validated: list[float] = []

    for value in value_list:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValidationError(
                f"Every value in {name} must be a real number."
            )

        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            raise ValidationError(
                f"Every value in {name} must be finite."
            )

        if positive and numeric_value <= 0.0:
            raise ValidationError(
                f"Every value in {name} must be positive."
            )

        if not positive and numeric_value < 0.0:
            raise ValidationError(
                f"Every value in {name} must be nonnegative."
            )

        validated.append(numeric_value)

    if len(set(validated)) != len(validated):
        raise ValidationError(
            f"{name} must not contain duplicate values."
        )

    return validated


def _validate_positive_integer(value: int, name: str) -> int:
    """Validate a positive integer experiment parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be a positive integer.")

    integer_value = int(value)

    if integer_value <= 0:
        raise ValidationError(f"{name} must be a positive integer.")

    return integer_value


def _validate_method_study_seeds(
    seeds: Sequence[int],
) -> list[int]:
    """Validate and copy unique method-study run seeds."""
    validated = _validate_seed_values(seeds)

    if len(set(validated)) != len(validated):
        raise ValidationError("seeds must not contain duplicates.")

    return validated


def _validate_method_study_sensor_counts(
    sensor_counts: Sequence[int],
    maximum: int,
) -> list[int]:
    """Validate and copy unique positive sensor counts."""
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

    validated: list[int] = []

    for count in count_list:
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise ValidationError(
                "Every sensor count must be a positive integer."
            )

        integer_count = int(count)

        if integer_count <= 0:
            raise ValidationError(
                "Every sensor count must be a positive integer."
            )

        if integer_count > maximum:
            raise ValidationError(
                "sensor counts cannot exceed the number of interior "
                f"grid points ({maximum})."
            )

        validated.append(integer_count)

    if len(set(validated)) != len(validated):
        raise ValidationError(
            "sensor_counts must not contain duplicates."
        )

    return validated


def _validate_sensor_strategies(
    strategies: Sequence[str],
) -> list[str]:
    """Validate and normalize sensor-layout study strategies."""
    if isinstance(strategies, (str, bytes)):
        raise ValidationError(
            "strategies must be a sequence of strategy names."
        )

    try:
        strategy_list = list(strategies)
    except TypeError as error:
        raise ValidationError(
            "strategies must be a sequence of strategy names."
        ) from error

    if not strategy_list:
        raise ValidationError(
            "strategies must contain at least one strategy."
        )

    supported = {"regular", "random", "center_focused"}
    normalized: list[str] = []

    for strategy in strategy_list:
        strategy_name = _normalize_choice(
            strategy,
            "strategy",
        )

        if strategy_name == "center":
            strategy_name = "center_focused"

        if strategy_name not in supported:
            raise ValidationError(
                "Unsupported strategy. Choose 'regular', 'random', "
                "or 'center_focused'."
            )

        normalized.append(strategy_name)

    if len(set(normalized)) != len(normalized):
        raise ValidationError(
            "strategies must not contain duplicates."
        )

    return normalized


def _study_seed(
    master_seed: int,
    run_seed: int,
    stream: int,
) -> int:
    """Derive one deterministic independent study seed."""
    seed_sequence = np.random.SeedSequence(
        [master_seed, run_seed, stream]
    )
    return int(
        seed_sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


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
    regularization: Literal["identity", "smoothness"] = "identity",
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
        regularization=regularization,
    )

    metrics = compute_all_metrics(
        true_source,
        reconstruction.source,
    )
    metrics.update(
        _measurement_metrics(
            reconstruction,
            sensor_data_noisy.values,
        )
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
        "regularization": regularization,
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
    regularization: Literal["identity", "smoothness"] = "identity",
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
    regularization:
        Identity or unconstrained first-difference regularization.

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

    validate_sensor_data_for_grid(sensor_data, grid)

    reconstruction = reconstruct_tikhonov(
        sensor_data,
        grid,
        alpha=alpha,
        regularization=regularization,
    )

    runtime = perf_counter() - start_time

    config: dict[str, Any] = {
        "mode": "user_measurements",
        "grid_shape": grid.shape,
        "domain_size": grid.domain.size,
        "num_sensors": len(sensor_data),
        "alpha": float(reconstruction.alpha),
        "regularization": regularization,
    }

    return MeasurementReconstructionResult(
        grid=grid,
        sensor_data=sensor_data,
        reconstruction=reconstruction,
        config=config,
        runtime=float(runtime),
    )


def reconstruct_from_temperature_field(
    temperature_field: ArrayLike,
    *,
    grid: Grid2D | None = None,
    domain: Domain2D | None = None,
    sensor_indices: Sequence[Sequence[Integral]] | None = None,
    alpha: Real = 1e-3,
    regularization: Literal["identity", "smoothness"] = "identity",
) -> MeasurementReconstructionResult:
    """Reconstruct a source from a supplied temperature field.

    By default, every interior temperature node is used as a
    measurement. A custom ordered subset may be supplied, but boundary
    nodes are intentionally excluded from this workflow.
    """
    if grid is not None and not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object or None.")

    if domain is not None and not isinstance(domain, Domain2D):
        raise ValidationError("domain must be a Domain2D object or None.")

    if grid is not None and domain is not None:
        raise ValidationError(
            "grid and domain must not both be supplied."
        )

    temperature = ensure_2d_array(
        temperature_field,
        name="temperature_field",
    )

    if temperature.shape[0] < 3 or temperature.shape[1] < 3:
        raise ValidationError(
            "temperature_field dimensions must each contain at least "
            "three grid points."
        )

    if grid is None:
        selected_domain = Domain2D() if domain is None else domain
        resolved_grid = Grid2D(
            nx=temperature.shape[0],
            ny=temperature.shape[1],
            domain=selected_domain,
        )
    else:
        resolved_grid = grid

        if temperature.shape != resolved_grid.shape:
            raise ValidationError(
                "temperature_field must have shape "
                f"{resolved_grid.shape}, but received "
                f"{temperature.shape}."
            )

    if sensor_indices is None:
        measurement_selection = "all_interior"
        indices = np.asarray(
            [
                (i, j)
                for i in range(1, resolved_grid.nx - 1)
                for j in range(1, resolved_grid.ny - 1)
            ],
            dtype=np.int64,
        )
    else:
        measurement_selection = "custom_indices"
        indices = _validate_interior_sensor_indices(
            sensor_indices,
            resolved_grid,
        )

    sensor_data = create_sensor_data(
        temperature,
        indices,
        resolved_grid,
    )
    measurement_result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=resolved_grid.shape,
        domain=resolved_grid.domain,
        alpha=alpha,
        regularization=regularization,
    )
    number_of_measurements = len(sensor_data)
    config: dict[str, Any] = {
        "mode": "temperature_field",
        "grid_shape": resolved_grid.shape,
        "domain_size": resolved_grid.domain.size,
        "num_sensors": number_of_measurements,
        "num_measurements": number_of_measurements,
        "measurement_selection": measurement_selection,
        "alpha": float(measurement_result.reconstruction.alpha),
        "regularization": measurement_result.config["regularization"],
    }

    return MeasurementReconstructionResult(
        grid=measurement_result.grid,
        sensor_data=measurement_result.sensor_data,
        reconstruction=measurement_result.reconstruction,
        config=config,
        runtime=measurement_result.runtime,
    )


def run_regularization_comparison(
    alpha_by_method: Mapping[str, Real],
    *,
    grid_shape: tuple[int, int] = (30, 30),
    domain: Domain2D | None = None,
    source_type: str = "two_gaussians",
    sensor_strategy: str = "regular",
    num_sensors: int = 25,
    noise_level: Real = 0.02,
    seed: int | None = 42,
) -> tuple[pd.DataFrame, list[ExperimentResult]]:
    """Compare identity and smoothness penalties on one benchmark.

    The two regularizers use caller-selected alpha values because their
    penalty operators have different numerical scales. Both inverse
    solves share the same source, temperature, sensor layout, and noisy
    measurements.
    """
    alpha_values = _validate_regularization_alphas(alpha_by_method)
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
    shared_start_time = perf_counter()
    grid = Grid2D(nx=nx, ny=ny, domain=selected_domain)
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
    shared_runtime = perf_counter() - shared_start_time

    source_name = _normalize_choice(source_type, "source_type")
    strategy_name = _normalize_choice(
        sensor_strategy,
        "sensor_strategy",
    )
    results: list[ExperimentResult] = []
    rows: list[dict[str, Any]] = []

    for method in ("identity", "smoothness"):
        alpha = alpha_values[method]
        reconstruction = reconstruct_tikhonov(
            sensor_data_noisy,
            grid,
            alpha=alpha,
            regularization=method,
        )
        metrics = compute_all_metrics(
            true_source,
            reconstruction.source,
        )
        metrics.update(
            _measurement_metrics(
                reconstruction,
                sensor_data_noisy.values,
            )
        )
        config: dict[str, Any] = {
            "mode": "synthetic_benchmark",
            "study_type": "regularization_comparison",
            "regularization": method,
            "alpha": alpha,
            "grid_shape": grid.shape,
            "domain_size": grid.domain.size,
            "source_type": source_name,
            "sensor_strategy": strategy_name,
            "num_sensors": int(num_sensors),
            "noise_level": noise_value,
            "seed": seed,
        }
        result = ExperimentResult(
            grid=grid,
            true_source=true_source,
            temperature=temperature,
            sensor_data_clean=sensor_data_clean,
            sensor_data_noisy=sensor_data_noisy,
            reconstruction=reconstruction,
            metrics=metrics,
            config=config,
            runtime=float(shared_runtime + reconstruction.runtime),
        )
        results.append(result)
        rows.append(
            {
                "regularization": method,
                "alpha": alpha,
                "sensor_count": reconstruction.n_sensors,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "relative_l2_error": metrics["relative_l2_error"],
                "max_absolute_error": metrics[
                    "max_absolute_error"
                ],
                "residual_norm": metrics["residual_norm"],
                "relative_residual": metrics["relative_residual"],
                "residual_rms": metrics["residual_rms"],
                "solution_norm": metrics["solution_norm"],
                "runtime_seconds": float(reconstruction.runtime),
            }
        )

    columns = [
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

    return pd.DataFrame(rows, columns=columns), results


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


def run_repeated_noise_study(
    noise_levels: Sequence[Real],
    seeds: Sequence[int],
    *,
    grid_shape: tuple[int, int] = (30, 30),
    domain: Domain2D | None = None,
    source_type: str = "two_gaussians",
    sensor_strategy: str = "regular",
    num_sensors: int = 25,
    alpha: Real = 1e-7,
    seed: int | None = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[ExperimentResult],
]:
    """Repeat the noise study using independent noise realizations.

    The source field and clean sensor layout are generated once from
    ``seed``. Each value in ``seeds`` is then used only for the
    measurement-noise realization. This separates noise variability
    from changes in the source or sensor geometry.

    Parameters
    ----------
    noise_levels:
        Nonnegative relative Gaussian noise levels.
    seeds:
        Nonnegative integer seeds for independent noise realizations.
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
        Master seed used only for the source and sensor geometry.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, list[ExperimentResult]]
        Long-form run table, summary table, and complete results.

    Notes
    -----
    Population standard deviation (``ddof=0``) is used so that a
    study with one seed reports zero variability instead of NaN.
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
    validated_seeds = _validate_seed_values(seeds)

    nx, ny = _validate_grid_shape(grid_shape)

    if domain is None:
        selected_domain = Domain2D()
    elif isinstance(domain, Domain2D):
        selected_domain = domain
    else:
        raise ValidationError(
            "domain must be a Domain2D object or None."
        )

    source_seed, sensor_seed, _ = _derived_seeds(seed)
    normalized_source = _normalize_choice(
        source_type,
        "source_type",
    )
    normalized_strategy = _normalize_choice(
        sensor_strategy,
        "sensor_strategy",
    )

    grid = Grid2D(
        nx=nx,
        ny=ny,
        domain=selected_domain,
    )
    true_source = _create_synthetic_source(
        grid,
        normalized_source,
        seed=source_seed,
    )
    temperature = solve_forward(true_source, grid)
    sensor_indices = _place_sensors(
        grid,
        normalized_strategy,
        num_sensors,
        seed=sensor_seed,
    )
    sensor_data_clean = create_sensor_data(
        temperature,
        sensor_indices,
        grid,
    )

    results: list[ExperimentResult] = []
    rows: list[dict[str, Any]] = []

    for noise_level in validated_levels:
        for noise_seed in validated_seeds:
            start_time = perf_counter()

            sensor_data_noisy = add_noise_to_sensor_data(
                sensor_data_clean,
                noise_level=noise_level,
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
            metrics.update(
                _measurement_metrics(
                    reconstruction,
                    sensor_data_noisy.values,
                )
            )

            runtime = perf_counter() - start_time
            measurement_difference = (
                sensor_data_noisy.values
                - sensor_data_clean.values
            )

            config: dict[str, Any] = {
                "mode": "synthetic_benchmark",
                "study_type": "repeated_noise",
                "grid_shape": grid.shape,
                "domain_size": grid.domain.size,
                "source_type": normalized_source,
                "sensor_strategy": normalized_strategy,
                "num_sensors": len(sensor_data_clean),
                "noise_level": noise_level,
                "alpha": float(reconstruction.alpha),
                "seed": seed,
                "noise_seed": noise_seed,
            }

            result = ExperimentResult(
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
            results.append(result)

            rows.append(
                {
                    "study_type": "repeated_noise",
                    "noise_level": noise_level,
                    "seed": noise_seed,
                    "measurement_noise_norm": float(
                        np.linalg.norm(measurement_difference)
                    ),
                    "mean_absolute_measurement_noise": float(
                        np.mean(np.abs(measurement_difference))
                    ),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "relative_l2_error": metrics[
                        "relative_l2_error"
                    ],
                    "max_absolute_error": metrics[
                        "max_absolute_error"
                    ],
                    "residual_norm": reconstruction.residual_norm,
                    "solution_norm": reconstruction.solution_norm,
                    "runtime": float(runtime),
                }
            )

    detailed = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []

    for noise_level in dict.fromkeys(validated_levels):
        selected = detailed[
            detailed["noise_level"] == noise_level
        ]

        summary_rows.append(
            {
                "noise_level": noise_level,
                "number_of_runs": len(selected),
                "mean_relative_l2_error": float(
                    selected["relative_l2_error"].mean()
                ),
                "std_relative_l2_error": float(
                    selected["relative_l2_error"].std(ddof=0)
                ),
                "mean_rmse": float(selected["rmse"].mean()),
                "std_rmse": float(
                    selected["rmse"].std(ddof=0)
                ),
                "mean_residual_norm": float(
                    selected["residual_norm"].mean()
                ),
                "std_residual_norm": float(
                    selected["residual_norm"].std(ddof=0)
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return detailed, summary, results


def run_compact_parameter_study(
    alphas: Sequence[Real],
    betas: Sequence[Real],
    *,
    grid_shape: tuple[int, int] = (30, 30),
    domain: Domain2D | None = None,
    source_type: str = "two_gaussians",
    sensor_strategy: str = "regular",
    num_sensors: int = 25,
    noise_level: Real = 0.02,
    seed: int | None = 42,
    near_zero_threshold: Real = 1e-8,
    max_iterations: int = 100_000,
    tolerance: Real = 1e-7,
) -> pd.DataFrame:
    """Evaluate compact reconstructions on one shared benchmark.

    Rows follow alpha-major Cartesian order. The regularization
    values are controlled study inputs rather than universally optimal
    choices.
    """
    alpha_values = _validate_compact_parameter_values(
        alphas,
        "alphas",
        positive=True,
    )
    beta_values = _validate_compact_parameter_values(
        betas,
        "betas",
        positive=False,
    )
    threshold_value = _validate_positive_real(
        near_zero_threshold,
        "near_zero_threshold",
    )
    iteration_limit = _validate_positive_integer(
        max_iterations,
        "max_iterations",
    )
    tolerance_value = _validate_positive_real(
        tolerance,
        "tolerance",
    )
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
    grid = Grid2D(nx=nx, ny=ny, domain=selected_domain)
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

    rows: list[dict[str, float | int]] = []
    interior_size = (grid.nx - 2) * (grid.ny - 2)

    for alpha in alpha_values:
        for beta in beta_values:
            reconstruction = reconstruct_compact_nonnegative(
                sensor_data_noisy,
                grid,
                alpha=alpha,
                beta=beta,
                max_iterations=iteration_limit,
                tolerance=tolerance_value,
            )
            source_metrics = compute_all_metrics(
                true_source,
                reconstruction.source,
            )
            measurement_metrics = _measurement_metrics(
                reconstruction,
                sensor_data_noisy.values,
            )
            interior_source = reconstruction.source[1:-1, 1:-1]
            x_differences = np.diff(
                interior_source,
                axis=0,
            ) / grid.dx
            y_differences = np.diff(
                interior_source,
                axis=1,
            ) / grid.dy
            gradient_norm = float(
                np.sqrt(
                    np.sum(x_differences**2)
                    + np.sum(y_differences**2)
                )
            )
            near_zero_count = int(
                np.count_nonzero(
                    np.abs(interior_source) <= threshold_value
                )
            )

            rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "relative_l2_error": source_metrics[
                        "relative_l2_error"
                    ],
                    "rmse": source_metrics["rmse"],
                    "mae": source_metrics["mae"],
                    "max_absolute_error": source_metrics[
                        "max_absolute_error"
                    ],
                    "residual_norm": measurement_metrics[
                        "residual_norm"
                    ],
                    "relative_residual": measurement_metrics[
                        "relative_residual"
                    ],
                    "residual_rms": measurement_metrics[
                        "residual_rms"
                    ],
                    "solution_norm": measurement_metrics[
                        "solution_norm"
                    ],
                    "gradient_norm": gradient_norm,
                    "near_zero_count": near_zero_count,
                    "near_zero_fraction": (
                        near_zero_count / interior_size
                    ),
                    "active_count": interior_size - near_zero_count,
                    "runtime_seconds": float(
                        reconstruction.runtime
                    ),
                }
            )

    columns = [
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

    return pd.DataFrame(rows, columns=columns)


def run_reconstruction_method_study(
    seeds: Sequence[int],
    sensor_counts: Sequence[int],
    *,
    grid_shape: tuple[int, int] = (30, 30),
    domain: Domain2D | None = None,
    source_type: str = "two_gaussians",
    noise_level: Real = 0.02,
    sensor_strategy: str = "regular",
    seed: int = 42,
    identity_alpha: Real = 1e-7,
    smooth_alpha: Real = 1e-9,
    compact_alpha: Real = 1e-9,
    compact_beta: Real = 1e-8,
    compact_max_iterations: int = 100_000,
    compact_tolerance: Real = 1e-7,
    near_zero_threshold: Real = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare three reconstruction methods on shared benchmarks.

    The regularization defaults are documented starting values, not
    universally optimal method settings. Population standard deviation
    (``ddof=0``) is used in the summary table.
    """
    validated_seeds = _validate_method_study_seeds(seeds)
    nx, ny = _validate_grid_shape(grid_shape)
    maximum_sensors = (nx - 2) * (ny - 2)
    validated_counts = _validate_method_study_sensor_counts(
        sensor_counts,
        maximum_sensors,
    )
    master_seed = _validate_seed_values([seed])[0]
    noise_value = _validate_nonnegative_real(
        noise_level,
        "noise_level",
    )
    identity_alpha_value = _validate_positive_real(
        identity_alpha,
        "identity_alpha",
    )
    smooth_alpha_value = _validate_positive_real(
        smooth_alpha,
        "smooth_alpha",
    )
    compact_alpha_value = _validate_positive_real(
        compact_alpha,
        "compact_alpha",
    )
    compact_beta_value = _validate_nonnegative_real(
        compact_beta,
        "compact_beta",
    )
    iteration_limit = _validate_positive_integer(
        compact_max_iterations,
        "compact_max_iterations",
    )
    tolerance_value = _validate_positive_real(
        compact_tolerance,
        "compact_tolerance",
    )
    threshold_value = _validate_positive_real(
        near_zero_threshold,
        "near_zero_threshold",
    )
    normalized_source = _normalize_choice(
        source_type,
        "source_type",
    )
    normalized_strategy = _normalize_choice(
        sensor_strategy,
        "sensor_strategy",
    )

    if domain is None:
        selected_domain = Domain2D()
    elif isinstance(domain, Domain2D):
        selected_domain = domain
    else:
        raise ValidationError(
            "domain must be a Domain2D object or None."
        )

    method_order = (
        "identity",
        "smooth_nonnegative",
        "compact_nonnegative",
    )
    rows: list[dict[str, Any]] = []

    for sensor_count in validated_counts:
        for run_seed in validated_seeds:
            grid = Grid2D(
                nx=nx,
                ny=ny,
                domain=selected_domain,
            )
            source_seed = _study_seed(
                master_seed,
                run_seed,
                401,
            )
            sensor_seed = _study_seed(
                master_seed,
                run_seed,
                500 + sensor_count,
            )
            noise_seed = _study_seed(
                master_seed,
                run_seed,
                900 + sensor_count,
            )
            true_source = _create_synthetic_source(
                grid,
                normalized_source,
                seed=source_seed,
            )
            temperature = solve_forward(true_source, grid)
            sensor_indices = _place_sensors(
                grid,
                normalized_strategy,
                sensor_count,
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

            reconstructions = (
                (
                    "identity",
                    reconstruct_tikhonov(
                        sensor_data_noisy,
                        grid,
                        alpha=identity_alpha_value,
                    ),
                ),
                (
                    "smooth_nonnegative",
                    reconstruct_smooth_tikhonov(
                        sensor_data_noisy,
                        grid,
                        alpha=smooth_alpha_value,
                        nonnegative=True,
                    ),
                ),
                (
                    "compact_nonnegative",
                    reconstruct_compact_nonnegative(
                        sensor_data_noisy,
                        grid,
                        alpha=compact_alpha_value,
                        beta=compact_beta_value,
                        max_iterations=iteration_limit,
                        tolerance=tolerance_value,
                    ),
                ),
            )
            interior_size = (grid.nx - 2) * (grid.ny - 2)

            for method, reconstruction in reconstructions:
                source_metrics = compute_all_metrics(
                    true_source,
                    reconstruction.source,
                )
                measurement_metrics = _measurement_metrics(
                    reconstruction,
                    sensor_data_noisy.values,
                )
                interior_source = reconstruction.source[
                    1:-1,
                    1:-1,
                ]
                x_differences = np.diff(
                    interior_source,
                    axis=0,
                ) / grid.dx
                y_differences = np.diff(
                    interior_source,
                    axis=1,
                ) / grid.dy
                gradient_norm = float(
                    np.sqrt(
                        np.sum(x_differences**2)
                        + np.sum(y_differences**2)
                    )
                )
                near_zero_count = int(
                    np.count_nonzero(
                        np.abs(interior_source)
                        <= threshold_value
                    )
                )

                rows.append(
                    {
                        "sensor_count": sensor_count,
                        "run_seed": run_seed,
                        "method": method,
                        "relative_l2_error": source_metrics[
                            "relative_l2_error"
                        ],
                        "rmse": source_metrics["rmse"],
                        "mae": source_metrics["mae"],
                        "max_absolute_error": source_metrics[
                            "max_absolute_error"
                        ],
                        "residual_norm": measurement_metrics[
                            "residual_norm"
                        ],
                        "relative_residual": measurement_metrics[
                            "relative_residual"
                        ],
                        "residual_rms": measurement_metrics[
                            "residual_rms"
                        ],
                        "solution_norm": measurement_metrics[
                            "solution_norm"
                        ],
                        "gradient_norm": gradient_norm,
                        "near_zero_count": near_zero_count,
                        "near_zero_fraction": (
                            near_zero_count / interior_size
                        ),
                        "active_count": (
                            interior_size - near_zero_count
                        ),
                        "source_min": float(
                            np.min(interior_source)
                        ),
                        "source_max": float(
                            np.max(interior_source)
                        ),
                        "runtime_seconds": float(
                            reconstruction.runtime
                        ),
                    }
                )

    detailed_columns = [
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
    detailed = pd.DataFrame(rows, columns=detailed_columns)
    summary_rows: list[dict[str, Any]] = []

    for sensor_count in validated_counts:
        for method in method_order:
            selected = detailed[
                (detailed["sensor_count"] == sensor_count)
                & (detailed["method"] == method)
            ]
            summary_rows.append(
                {
                    "sensor_count": sensor_count,
                    "method": method,
                    "number_of_runs": len(selected),
                    "mean_relative_l2_error": float(
                        selected["relative_l2_error"].mean()
                    ),
                    "std_relative_l2_error": float(
                        selected["relative_l2_error"].std(ddof=0)
                    ),
                    "mean_rmse": float(selected["rmse"].mean()),
                    "std_rmse": float(
                        selected["rmse"].std(ddof=0)
                    ),
                    "mean_residual_norm": float(
                        selected["residual_norm"].mean()
                    ),
                    "std_residual_norm": float(
                        selected["residual_norm"].std(ddof=0)
                    ),
                    "mean_near_zero_fraction": float(
                        selected["near_zero_fraction"].mean()
                    ),
                    "std_near_zero_fraction": float(
                        selected["near_zero_fraction"].std(ddof=0)
                    ),
                    "mean_runtime_seconds": float(
                        selected["runtime_seconds"].mean()
                    ),
                }
            )

    summary_columns = [
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
    summary = pd.DataFrame(
        summary_rows,
        columns=summary_columns,
    )

    return detailed, summary


def run_sensor_layout_study(
    strategies: Sequence[str],
    seeds: Sequence[int],
    *,
    grid_shape: tuple[int, int] = (30, 30),
    domain: Domain2D | None = None,
    source_type: str = "two_gaussians",
    num_sensors: int = 25,
    noise_level: Real = 0.02,
    alpha: Real = 1e-7,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[ExperimentResult],
]:
    """Compare sensor-placement strategies under shared conditions.

    The true source and temperature field are fixed for all strategies.
    Regular and center-focused layouts remain fixed across repeated
    runs, while random layouts change with the run seed. Each run seed
    also determines an independent, reproducible noise realization.

    Population standard deviation (``ddof=0``) is used so that a
    single-run strategy reports zero observed variability.
    """
    normalized_strategies = _validate_sensor_strategies(strategies)
    validated_seeds = _validate_seed_values(seeds)
    master_seed = _validate_seed_values([seed])[0]
    nx, ny = _validate_grid_shape(grid_shape)
    noise_value = _validate_nonnegative_real(
        noise_level,
        "noise_level",
    )
    alpha_value = _validate_positive_real(alpha, "alpha")

    if (
        isinstance(num_sensors, bool)
        or not isinstance(num_sensors, Integral)
    ):
        raise ValidationError("num_sensors must be an integer.")

    sensor_count = int(num_sensors)
    maximum_sensors = (nx - 2) * (ny - 2)

    if sensor_count <= 0:
        raise ValidationError(
            "num_sensors must be greater than zero."
        )

    if sensor_count > maximum_sensors:
        raise ValidationError(
            "num_sensors cannot exceed the number of interior "
            f"grid points ({maximum_sensors})."
        )

    if domain is None:
        selected_domain = Domain2D()
    elif isinstance(domain, Domain2D):
        selected_domain = domain
    else:
        raise ValidationError(
            "domain must be a Domain2D object or None."
        )

    source_seed, fixed_layout_seed, _ = _derived_seeds(
        master_seed
    )
    normalized_source = _normalize_choice(
        source_type,
        "source_type",
    )

    grid = Grid2D(
        nx=nx,
        ny=ny,
        domain=selected_domain,
    )
    true_source = _create_synthetic_source(
        grid,
        normalized_source,
        seed=source_seed,
    )
    temperature = solve_forward(true_source, grid)

    fixed_sensor_data: dict[str, SensorData] = {}

    for strategy in normalized_strategies:
        if strategy == "random":
            continue

        indices = _place_sensors(
            grid,
            strategy,
            sensor_count,
            seed=fixed_layout_seed,
        )
        fixed_sensor_data[strategy] = create_sensor_data(
            temperature,
            indices,
            grid,
        )

    rows: list[dict[str, Any]] = []
    results: list[ExperimentResult] = []

    for strategy in normalized_strategies:
        for run_seed in validated_seeds:
            start_time = perf_counter()

            if strategy == "random":
                layout_seed = _study_seed(
                    master_seed,
                    run_seed,
                    101,
                )
                sensor_indices = _place_sensors(
                    grid,
                    strategy,
                    sensor_count,
                    seed=layout_seed,
                )
                sensor_data_clean = create_sensor_data(
                    temperature,
                    sensor_indices,
                    grid,
                )
            else:
                layout_seed = fixed_layout_seed
                sensor_data_clean = fixed_sensor_data[strategy]

            noise_seed = _study_seed(
                master_seed,
                run_seed,
                202,
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
                alpha=alpha_value,
            )
            metrics = compute_all_metrics(
                true_source,
                reconstruction.source,
            )
            metrics.update(
                _measurement_metrics(
                    reconstruction,
                    sensor_data_noisy.values,
                )
            )

            total_runtime = perf_counter() - start_time

            config: dict[str, Any] = {
                "mode": "synthetic_benchmark",
                "study_type": "sensor_layout",
                "grid_shape": grid.shape,
                "domain_size": grid.domain.size,
                "source_type": normalized_source,
                "sensor_strategy": strategy,
                "num_sensors": len(sensor_data_clean),
                "noise_level": noise_value,
                "alpha": float(reconstruction.alpha),
                "seed": master_seed,
                "run_seed": run_seed,
                "layout_seed": layout_seed,
                "noise_seed": noise_seed,
            }

            result = ExperimentResult(
                grid=grid,
                true_source=true_source,
                temperature=temperature,
                sensor_data_clean=sensor_data_clean,
                sensor_data_noisy=sensor_data_noisy,
                reconstruction=reconstruction,
                metrics=metrics,
                config=config,
                runtime=float(total_runtime),
            )
            results.append(result)

            rows.append(
                {
                    "strategy": strategy,
                    "run_seed": run_seed,
                    "sensor_count": len(sensor_data_clean),
                    "relative_l2_error": metrics[
                        "relative_l2_error"
                    ],
                    "rmse": metrics["rmse"],
                    "residual_norm": metrics["residual_norm"],
                    "relative_residual": metrics[
                        "relative_residual"
                    ],
                    "residual_rms": metrics["residual_rms"],
                    "runtime_seconds": float(
                        reconstruction.runtime
                    ),
                }
            )

    detailed = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []

    for strategy in normalized_strategies:
        selected = detailed[detailed["strategy"] == strategy]

        summary_rows.append(
            {
                "strategy": strategy,
                "number_of_runs": len(selected),
                "mean_relative_l2_error": float(
                    selected["relative_l2_error"].mean()
                ),
                "std_relative_l2_error": float(
                    selected["relative_l2_error"].std(ddof=0)
                ),
                "mean_rmse": float(selected["rmse"].mean()),
                "std_rmse": float(
                    selected["rmse"].std(ddof=0)
                ),
                "mean_residual_norm": float(
                    selected["residual_norm"].mean()
                ),
                "std_residual_norm": float(
                    selected["residual_norm"].std(ddof=0)
                ),
                "mean_runtime_seconds": float(
                    selected["runtime_seconds"].mean()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return detailed, summary, results
