"""Scientific visualization tools for ThermoReconLab.

This module contains a small set of reusable Matplotlib functions for
two-dimensional fields, reconstruction errors, and sparse sensors.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from thermoreconlab.core.fields import ensure_2d_array, validate_field
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.sensors import SensorData, custom_sensors


DEFAULT_FIGURE_SIZE: Final[tuple[float, float]] = (6.5, 5.0)


def _prepare_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return an existing axis or create a new figure and axis."""
    if ax is None:
        figure, new_axis = plt.subplots(
            figsize=DEFAULT_FIGURE_SIZE
        )
        return figure, new_axis

    if not isinstance(ax, Axes):
        raise ValidationError(
            "ax must be a matplotlib Axes object or None."
        )

    return ax.figure, ax


def _validate_color_limits(
    vmin: float | None,
    vmax: float | None,
) -> tuple[float | None, float | None]:
    """Validate optional explicit color limits."""
    validated: list[float | None] = []

    for name, value in (("vmin", vmin), ("vmax", vmax)):
        if value is None:
            validated.append(None)
            continue

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise ValidationError(
                f"{name} must be a finite real number or None."
            )

        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            raise ValidationError(
                f"{name} must be a finite real number or None."
            )

        validated.append(numeric_value)

    lower, upper = validated

    if (
        lower is not None
        and upper is not None
        and lower >= upper
    ):
        raise ValidationError("vmin must be smaller than vmax.")

    return lower, upper


def plot_heatmap(
    field: ArrayLike,
    *,
    grid: Grid2D | None = None,
    title: str = "",
    colorbar_label: str = "",
    ax: Axes | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Figure, Axes]:
    """Plot a two-dimensional scalar field."""
    if grid is None:
        field_array = ensure_2d_array(field)
    elif isinstance(grid, Grid2D):
        field_array = validate_field(field, grid)
    else:
        raise ValidationError(
            "grid must be a Grid2D object or None."
        )

    if not isinstance(title, str):
        raise ValidationError("title must be a string.")

    if not isinstance(colorbar_label, str):
        raise ValidationError(
            "colorbar_label must be a string."
        )

    if not isinstance(cmap, str) or not cmap.strip():
        raise ValidationError(
            "cmap must be a non-empty string."
        )

    lower_limit, upper_limit = _validate_color_limits(
        vmin,
        vmax,
    )
    figure, axis = _prepare_axes(ax)

    if grid is None:
        image = axis.imshow(
            field_array.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=lower_limit,
            vmax=upper_limit,
        )
        axis.set_xlabel("i index")
        axis.set_ylabel("j index")
    else:
        image = axis.pcolormesh(
            grid.X,
            grid.Y,
            field_array,
            shading="auto",
            cmap=cmap,
            vmin=lower_limit,
            vmax=upper_limit,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")

    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis)

    if colorbar_label:
        colorbar.set_label(colorbar_label)

    return figure, axis


def plot_source(
    source: ArrayLike,
    *,
    grid: Grid2D | None = None,
    title: str = "Heat-source field",
    ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Figure, Axes]:
    """Plot a heat-source field."""
    return plot_heatmap(
        source,
        grid=grid,
        title=title,
        colorbar_label="Source intensity",
        ax=ax,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )


def plot_temperature(
    temperature: ArrayLike,
    *,
    grid: Grid2D | None = None,
    title: str = "Temperature field",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a temperature field."""
    return plot_heatmap(
        temperature,
        grid=grid,
        title=title,
        colorbar_label="Temperature",
        ax=ax,
        cmap="viridis",
    )


def plot_error_map(
    error_field: ArrayLike,
    *,
    grid: Grid2D | None = None,
    title: str = "Reconstruction error",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a signed source-reconstruction error field."""
    if grid is None:
        error_array = ensure_2d_array(error_field)
    elif isinstance(grid, Grid2D):
        error_array = validate_field(error_field, grid)
    else:
        raise ValidationError(
            "grid must be a Grid2D object or None."
        )

    limit = float(np.max(np.abs(error_array)))

    if limit == 0.0:
        limit = 1.0

    figure, axis = _prepare_axes(ax)

    if grid is None:
        image = axis.imshow(
            error_array.T,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_xlabel("i index")
        axis.set_ylabel("j index")
    else:
        image = axis.pcolormesh(
            grid.X,
            grid.Y,
            error_array,
            shading="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")

    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Signed error")

    return figure, axis


def plot_sensor_layout(
    sensor_indices: ArrayLike,
    grid: Grid2D,
    *,
    background: ArrayLike | None = None,
    title: str = "Sensor layout",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot sensor locations, optionally over a background field."""
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    indices = custom_sensors(sensor_indices, grid)
    figure, axis = _prepare_axes(ax)

    if background is not None:
        background_array = validate_field(
            background,
            grid,
            name="background",
        )
        image = axis.pcolormesh(
            grid.X,
            grid.Y,
            background_array,
            shading="auto",
            cmap="viridis",
        )
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("Field value")

    x_coordinates = grid.x[indices[:, 0]]
    y_coordinates = grid.y[indices[:, 1]]

    axis.scatter(
        x_coordinates,
        y_coordinates,
        marker="o",
        edgecolors="black",
        linewidths=0.8,
    )

    axis.set_xlim(0.0, grid.domain.length_x)
    axis.set_ylim(0.0, grid.domain.length_y)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")
    axis.set_title(title)

    return figure, axis


def plot_sensor_measurements(
    sensor_data: SensorData,
    grid: Grid2D,
    *,
    background: ArrayLike | None = None,
    title: str = "Sensor measurements",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot sensor positions colored by measured temperature."""
    if not isinstance(sensor_data, SensorData):
        raise ValidationError(
            "sensor_data must be a SensorData object."
        )

    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    indices = custom_sensors(sensor_data.indices, grid)
    figure, axis = _prepare_axes(ax)

    if background is not None:
        background_array = validate_field(
            background,
            grid,
            name="background",
        )
        background_image = axis.pcolormesh(
            grid.X,
            grid.Y,
            background_array,
            shading="auto",
            cmap="Greys",
        )
        background_colorbar = figure.colorbar(
            background_image,
            ax=axis,
        )
        background_colorbar.set_label("Background field")

    x_coordinates = grid.x[indices[:, 0]]
    y_coordinates = grid.y[indices[:, 1]]

    scatter = axis.scatter(
        x_coordinates,
        y_coordinates,
        c=sensor_data.values,
        marker="o",
        edgecolors="black",
        linewidths=0.8,
        cmap="viridis",
    )

    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Measured temperature")

    axis.set_xlim(0.0, grid.domain.length_x)
    axis.set_ylim(0.0, grid.domain.length_y)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")
    axis.set_title(title)

    return figure, axis


def plot_regularization_study(
    dataframe: pd.DataFrame,
    *,
    metric: str = "relative_l2_error",
    title: str = "Regularization parameter study",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot reconstruction error against the regularization parameter.

    Parameters
    ----------
    dataframe:
        Study results returned by ``run_regularization_study``.
    metric:
        DataFrame column to display on the vertical axis.
    title:
        Plot title.
    ax:
        Optional Matplotlib axis.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axis containing the plot.

    Raises
    ------
    ValidationError
        If the DataFrame is empty, required columns are missing, or
        alpha and metric values are invalid.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValidationError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValidationError(
            "dataframe must contain at least one result."
        )

    if not isinstance(metric, str) or not metric.strip():
        raise ValidationError(
            "metric must be a non-empty string."
        )

    required_columns = {"alpha", metric}
    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValidationError(
            "Regularization study data is missing columns: "
            f"{sorted(missing_columns)}."
        )

    try:
        alpha_values = dataframe["alpha"].to_numpy(
            dtype=float
        )
        metric_values = dataframe[metric].to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "alpha and metric columns must contain numeric values."
        ) from error

    if not np.all(np.isfinite(alpha_values)):
        raise ValidationError(
            "alpha values must be finite."
        )

    if np.any(alpha_values <= 0.0):
        raise ValidationError(
            "alpha values must be greater than zero."
        )

    if not np.all(np.isfinite(metric_values)):
        raise ValidationError(
            f"{metric} values must be finite."
        )

    order = np.argsort(alpha_values)
    alpha_values = alpha_values[order]
    metric_values = metric_values[order]

    figure, axis = _prepare_axes(ax)

    axis.plot(
        alpha_values,
        metric_values,
        marker="o",
    )

    best_index = int(np.argmin(metric_values))
    best_alpha = alpha_values[best_index]
    best_metric = metric_values[best_index]

    axis.scatter(
        [best_alpha],
        [best_metric],
        marker="*",
        s=140,
        label=f"Best α = {best_alpha:.1e}",
    )

    axis.set_xscale("log")
    axis.set_xlabel("Regularization parameter α")
    axis.set_ylabel(
        metric.replace("_", " ").title()
    )
    axis.set_title(title)
    axis.grid(True, which="both", linestyle=":")
    axis.legend()

    return figure, axis


def plot_sensor_count_study(
    dataframe: pd.DataFrame,
    *,
    metric: str = "relative_l2_error",
    title: str = "Sensor-count study",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot reconstruction quality against the number of sensors.

    Parameters
    ----------
    dataframe:
        Results returned by ``run_sensor_count_study``.
    metric:
        DataFrame column displayed on the vertical axis.
    title:
        Plot title.
    ax:
        Optional existing Matplotlib axis.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axis containing the study plot.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValidationError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValidationError(
            "dataframe must contain at least one result."
        )

    if not isinstance(metric, str) or not metric.strip():
        raise ValidationError(
            "metric must be a non-empty string."
        )

    required_columns = {"sensor_count", metric}
    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValidationError(
            "Sensor-count study data is missing columns: "
            f"{sorted(missing_columns)}."
        )

    try:
        sensor_counts = dataframe["sensor_count"].to_numpy(
            dtype=float
        )
        metric_values = dataframe[metric].to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "sensor_count and metric columns must be numeric."
        ) from error

    if not np.all(np.isfinite(sensor_counts)):
        raise ValidationError(
            "sensor_count values must be finite."
        )

    if np.any(sensor_counts <= 0.0):
        raise ValidationError(
            "sensor_count values must be greater than zero."
        )

    if not np.allclose(sensor_counts, np.round(sensor_counts)):
        raise ValidationError(
            "sensor_count values must be integers."
        )

    if not np.all(np.isfinite(metric_values)):
        raise ValidationError(
            f"{metric} values must be finite."
        )

    order = np.argsort(sensor_counts)
    sensor_counts = sensor_counts[order]
    metric_values = metric_values[order]

    figure, axis = _prepare_axes(ax)

    axis.plot(
        sensor_counts,
        metric_values,
        marker="o",
    )

    best_index = int(np.argmin(metric_values))
    best_count = int(sensor_counts[best_index])
    best_metric = metric_values[best_index]

    axis.scatter(
        [best_count],
        [best_metric],
        marker="*",
        s=140,
        label=f"Best count = {best_count}",
    )

    axis.set_xlabel("Number of sensors")
    axis.set_ylabel(
        metric.replace("_", " ").title()
    )
    axis.set_title(title)
    axis.grid(True, linestyle=":")
    axis.legend()

    return figure, axis


def plot_noise_sensitivity_study(
    dataframe: pd.DataFrame,
    *,
    metric: str = "relative_l2_error",
    title: str = "Noise-sensitivity study",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot reconstruction quality against measurement-noise level.

    Parameters
    ----------
    dataframe:
        Results returned by ``run_noise_sensitivity_study``.
    metric:
        DataFrame column displayed on the vertical axis.
    title:
        Plot title.
    ax:
        Optional existing Matplotlib axis.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axis containing the study plot.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValidationError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValidationError(
            "dataframe must contain at least one result."
        )

    if not isinstance(metric, str) or not metric.strip():
        raise ValidationError(
            "metric must be a non-empty string."
        )

    required_columns = {"noise_level", metric}
    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValidationError(
            "Noise-sensitivity study data is missing columns: "
            f"{sorted(missing_columns)}."
        )

    try:
        noise_levels = dataframe["noise_level"].to_numpy(
            dtype=float
        )
        metric_values = dataframe[metric].to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "noise_level and metric columns must be numeric."
        ) from error

    if not np.all(np.isfinite(noise_levels)):
        raise ValidationError(
            "noise_level values must be finite."
        )

    if np.any(noise_levels < 0.0):
        raise ValidationError(
            "noise_level values must be nonnegative."
        )

    if not np.all(np.isfinite(metric_values)):
        raise ValidationError(
            f"{metric} values must be finite."
        )

    order = np.argsort(noise_levels)
    noise_levels = noise_levels[order]
    metric_values = metric_values[order]

    figure, axis = _prepare_axes(ax)

    axis.plot(
        noise_levels,
        metric_values,
        marker="o",
    )

    best_index = int(np.argmin(metric_values))
    best_noise = noise_levels[best_index]
    best_metric = metric_values[best_index]

    axis.scatter(
        [best_noise],
        [best_metric],
        marker="*",
        s=140,
        label=f"Lowest error at noise = {best_noise:.1%}",
    )

    axis.set_xlabel("Relative measurement-noise level")
    axis.set_ylabel(
        metric.replace("_", " ").title()
    )
    axis.set_title(title)
    axis.grid(True, linestyle=":")
    axis.legend()

    return figure, axis


def plot_repeated_noise_study(
    dataframe: pd.DataFrame,
    *,
    title: str = "Repeated-noise study",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot mean source error and between-run variability by noise level.

    The input must be the summary table returned by
    ``run_repeated_noise_study``. Error bars show the population
    standard deviation of the relative L2 source error.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValidationError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValidationError(
            "dataframe must contain at least one summary row."
        )

    required_columns = {
        "noise_level",
        "mean_relative_l2_error",
        "std_relative_l2_error",
    }
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValidationError(
            "Repeated-noise summary is missing columns: "
            f"{sorted(missing_columns)}."
        )

    try:
        noise_levels = dataframe["noise_level"].to_numpy(
            dtype=float,
            copy=True,
        )
        mean_errors = dataframe[
            "mean_relative_l2_error"
        ].to_numpy(dtype=float, copy=True)
        standard_deviations = dataframe[
            "std_relative_l2_error"
        ].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "Repeated-noise summary columns must be numeric."
        ) from error

    if not np.all(np.isfinite(noise_levels)):
        raise ValidationError("noise_level values must be finite.")

    if not np.all(np.isfinite(mean_errors)):
        raise ValidationError(
            "mean_relative_l2_error values must be finite."
        )

    if not np.all(np.isfinite(standard_deviations)):
        raise ValidationError(
            "std_relative_l2_error values must be finite."
        )

    if np.any(noise_levels < 0.0):
        raise ValidationError(
            "noise_level values must be nonnegative."
        )

    if np.any(mean_errors < 0.0):
        raise ValidationError(
            "mean_relative_l2_error values must be nonnegative."
        )

    if np.any(standard_deviations < 0.0):
        raise ValidationError(
            "std_relative_l2_error values must be nonnegative."
        )

    if np.unique(noise_levels).size != noise_levels.size:
        raise ValidationError(
            "Repeated-noise summary must contain one row per "
            "noise level."
        )

    order = np.argsort(noise_levels, kind="stable")
    sorted_noise = noise_levels[order]
    sorted_means = mean_errors[order]
    sorted_standard_deviations = standard_deviations[order]

    figure, axis = _prepare_axes(ax)

    axis.errorbar(
        sorted_noise,
        sorted_means,
        yerr=sorted_standard_deviations,
        fmt="o-",
        capsize=4,
    )

    axis.set_xlabel("Relative measurement-noise level")
    axis.set_ylabel("Mean relative L2 source error")
    axis.set_title(title)
    axis.grid(True, linestyle=":")

    return figure, axis


def plot_sensor_layout_study(
    dataframe: pd.DataFrame,
    *,
    title: str = "Sensor-layout comparison",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot mean source error and variability for sensor strategies."""
    if not isinstance(dataframe, pd.DataFrame):
        raise ValidationError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValidationError(
            "dataframe must contain at least one summary row."
        )

    required_columns = {
        "strategy",
        "mean_relative_l2_error",
        "std_relative_l2_error",
    }
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValidationError(
            "Sensor-layout summary is missing columns: "
            f"{sorted(missing_columns)}."
        )

    strategies = dataframe["strategy"].tolist()

    if any(
        not isinstance(strategy, str) or not strategy.strip()
        for strategy in strategies
    ):
        raise ValidationError(
            "strategy values must be non-empty strings."
        )

    normalized_strategies = [
        strategy.strip()
        for strategy in strategies
    ]

    if len(set(normalized_strategies)) != len(
        normalized_strategies
    ):
        raise ValidationError(
            "Sensor-layout summary must contain one row per strategy."
        )

    try:
        mean_errors = dataframe[
            "mean_relative_l2_error"
        ].to_numpy(dtype=float, copy=True)
        standard_deviations = dataframe[
            "std_relative_l2_error"
        ].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "Sensor-layout error columns must be numeric."
        ) from error

    if not np.all(np.isfinite(mean_errors)):
        raise ValidationError(
            "mean_relative_l2_error values must be finite."
        )

    if not np.all(np.isfinite(standard_deviations)):
        raise ValidationError(
            "std_relative_l2_error values must be finite."
        )

    if np.any(mean_errors < 0.0):
        raise ValidationError(
            "mean_relative_l2_error values must be nonnegative."
        )

    if np.any(standard_deviations < 0.0):
        raise ValidationError(
            "std_relative_l2_error values must be nonnegative."
        )

    positions = np.arange(len(normalized_strategies))
    labels = [
        strategy.replace("_", " ").title()
        for strategy in normalized_strategies
    ]

    figure, axis = _prepare_axes(ax)

    axis.bar(
        positions,
        mean_errors,
        yerr=standard_deviations,
        capsize=4,
    )

    axis.set_xticks(positions, labels)
    axis.set_xlabel("Sensor strategy")
    axis.set_ylabel("Mean relative L2 source error")
    axis.set_title(title)
    axis.grid(True, axis="y", linestyle=":")

    return figure, axis


def plot_reconstruction_comparison(
    grid: Grid2D,
    true_source: ArrayLike,
    temperature: ArrayLike,
    sensor_data: SensorData,
    reconstructed_source: ArrayLike,
    *,
    source_vmin: float | None = None,
    source_vmax: float | None = None,
    title: str = "Reconstruction comparison",
) -> tuple[Figure, np.ndarray]:
    """Create a fair 2-by-2 reconstruction comparison figure."""
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    if not isinstance(sensor_data, SensorData):
        raise ValidationError(
            "sensor_data must be a SensorData object."
        )

    if not isinstance(title, str):
        raise ValidationError("title must be a string.")

    true_array = validate_field(
        true_source,
        grid,
        name="true_source",
    )
    temperature_array = validate_field(
        temperature,
        grid,
        name="temperature",
    )
    reconstructed_array = validate_field(
        reconstructed_source,
        grid,
        name="reconstructed_source",
    )
    indices = custom_sensors(sensor_data.indices, grid)

    explicit_min, explicit_max = _validate_color_limits(
        source_vmin,
        source_vmax,
    )

    joint_min = float(
        min(np.min(true_array), np.min(reconstructed_array))
    )
    joint_max = float(
        max(np.max(true_array), np.max(reconstructed_array))
    )

    lower_limit = (
        joint_min if explicit_min is None else explicit_min
    )
    upper_limit = (
        joint_max if explicit_max is None else explicit_max
    )

    if lower_limit >= upper_limit:
        if explicit_min is not None or explicit_max is not None:
            raise ValidationError(
                "Source color limits must enclose a nonzero range."
            )

        padding = (
            1.0
            if lower_limit == 0.0
            else abs(lower_limit) * 0.05
        )
        lower_limit -= padding
        upper_limit += padding

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 9.0),
        constrained_layout=True,
    )

    plot_source(
        true_array,
        grid=grid,
        title="True source",
        ax=axes[0, 0],
        vmin=lower_limit,
        vmax=upper_limit,
    )

    plot_temperature(
        temperature_array,
        grid=grid,
        title="Temperature and sensors",
        ax=axes[0, 1],
    )
    axes[0, 1].scatter(
        grid.x[indices[:, 0]],
        grid.y[indices[:, 1]],
        marker="o",
        facecolors="none",
        edgecolors="white",
        linewidths=1.2,
    )

    plot_source(
        reconstructed_array,
        grid=grid,
        title="Reconstructed source",
        ax=axes[1, 0],
        vmin=lower_limit,
        vmax=upper_limit,
    )

    error_field = reconstructed_array - true_array
    plot_error_map(
        error_field,
        grid=grid,
        title="Signed source error",
        ax=axes[1, 1],
    )

    figure.suptitle(title)

    return figure, axes


def plot_reconstruction_method_comparison(
    grid: Grid2D,
    true_source: ArrayLike,
    temperature: ArrayLike,
    sensor_data: SensorData,
    identity_source: ArrayLike,
    smooth_source: ArrayLike,
    compact_source: ArrayLike,
    *,
    title: str = "Reconstruction-method comparison",
    source_vmin: float | None = None,
    source_vmax: float | None = None,
    error_limit: float | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot a fair 2-by-4 reconstruction-method comparison."""
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")

    if not isinstance(sensor_data, SensorData):
        raise ValidationError(
            "sensor_data must be a SensorData object."
        )

    if not isinstance(title, str):
        raise ValidationError("title must be a string.")

    true_array = validate_field(
        true_source,
        grid,
        name="true_source",
    )
    temperature_array = validate_field(
        temperature,
        grid,
        name="temperature",
    )
    identity_array = validate_field(
        identity_source,
        grid,
        name="identity_source",
    )
    smooth_array = validate_field(
        smooth_source,
        grid,
        name="smooth_source",
    )
    compact_array = validate_field(
        compact_source,
        grid,
        name="compact_source",
    )
    indices = custom_sensors(sensor_data.indices, grid)

    explicit_min, explicit_max = _validate_color_limits(
        source_vmin,
        source_vmax,
    )
    source_arrays = (
        true_array,
        identity_array,
        smooth_array,
        compact_array,
    )
    joint_min = float(
        min(np.min(field) for field in source_arrays)
    )
    joint_max = float(
        max(np.max(field) for field in source_arrays)
    )
    lower_limit = (
        joint_min if explicit_min is None else explicit_min
    )
    upper_limit = (
        joint_max if explicit_max is None else explicit_max
    )

    if lower_limit >= upper_limit:
        reference = max(
            abs(lower_limit),
            abs(upper_limit),
            1.0,
        )
        padding = 0.05 * reference

        if explicit_min is not None:
            upper_limit = lower_limit + padding
        elif explicit_max is not None:
            lower_limit = upper_limit - padding
        else:
            lower_limit -= padding
            upper_limit += padding

    if error_limit is None:
        error_limit_value = float(
            max(
                np.max(np.abs(field - true_array))
                for field in (
                    identity_array,
                    smooth_array,
                    compact_array,
                )
            )
        )

        if error_limit_value == 0.0:
            error_limit_value = 1.0
    else:
        if isinstance(error_limit, bool) or not isinstance(
            error_limit,
            (int, float, np.integer, np.floating),
        ):
            raise ValidationError(
                "error_limit must be a finite positive real number."
            )

        error_limit_value = float(error_limit)

        if (
            not np.isfinite(error_limit_value)
            or error_limit_value <= 0.0
        ):
            raise ValidationError(
                "error_limit must be a finite positive real number."
            )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(18.0, 8.5),
        constrained_layout=True,
    )
    source_titles = (
        "(a) True source",
        "(b) Identity Tikhonov",
        "(c) Smooth nonnegative",
        "(d) Compact nonnegative",
    )
    source_images = []

    for axis, field, panel_title in zip(
        axes[0],
        source_arrays,
        source_titles,
    ):
        image = axis.pcolormesh(
            grid.X,
            grid.Y,
            field,
            shading="auto",
            cmap="inferno",
            vmin=lower_limit,
            vmax=upper_limit,
        )
        source_images.append(image)
        axis.set_title(panel_title, fontsize=11)

    source_colorbar = figure.colorbar(
        source_images[0],
        ax=axes[0, :].tolist(),
        shrink=0.88,
        pad=0.02,
    )
    source_colorbar.set_label("Source intensity")

    temperature_image = axes[1, 0].pcolormesh(
        grid.X,
        grid.Y,
        temperature_array,
        shading="auto",
        cmap="viridis",
    )
    axes[1, 0].scatter(
        grid.x[indices[:, 0]],
        grid.y[indices[:, 1]],
        s=44,
        marker="o",
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=3,
    )
    axes[1, 0].set_title(
        "(e) Temperature and sensors",
        fontsize=11,
    )
    temperature_colorbar = figure.colorbar(
        temperature_image,
        ax=axes[1, 0],
        shrink=0.88,
        pad=0.02,
    )
    temperature_colorbar.set_label("Temperature")

    error_arrays = (
        identity_array - true_array,
        smooth_array - true_array,
        compact_array - true_array,
    )
    error_titles = (
        "(f) Identity signed error",
        "(g) Smooth signed error",
        "(h) Compact signed error",
    )
    error_images = []

    for axis, field, panel_title in zip(
        axes[1, 1:],
        error_arrays,
        error_titles,
    ):
        image = axis.pcolormesh(
            grid.X,
            grid.Y,
            field,
            shading="auto",
            cmap="coolwarm",
            vmin=-error_limit_value,
            vmax=error_limit_value,
        )
        error_images.append(image)
        axis.set_title(panel_title, fontsize=11)

    error_colorbar = figure.colorbar(
        error_images[0],
        ax=axes[1, 1:].tolist(),
        shrink=0.88,
        pad=0.02,
    )
    error_colorbar.set_label("Signed source error")

    for axis in axes.ravel():
        axis.set_xlim(0.0, grid.domain.length_x)
        axis.set_ylim(0.0, grid.domain.length_y)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        axis.tick_params(labelsize=9)

    figure.suptitle(title, fontsize=15)

    return figure, axes


def plot_singular_values(
    singular_values: ArrayLike,
    *,
    rank_tolerance: Real | None = None,
    numerical_rank: Integral | None = None,
    title: str = "Observation-matrix singular values",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot singular values in descending index order."""
    try:
        values = np.asarray(singular_values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "singular_values must contain numeric values."
        ) from error

    if values.ndim != 1:
        raise ValidationError(
            "singular_values must be one-dimensional."
        )

    if values.size == 0:
        raise ValidationError(
            "singular_values must not be empty."
        )

    if not np.all(np.isfinite(values)):
        raise ValidationError(
            "singular_values must contain only finite values."
        )

    if np.any(values < 0.0):
        raise ValidationError(
            "singular_values must be nonnegative."
        )

    if not isinstance(title, str):
        raise ValidationError("title must be a string.")

    tolerance_value: float | None = None

    if rank_tolerance is not None:
        if (
            isinstance(rank_tolerance, (bool, np.bool_))
            or not isinstance(rank_tolerance, Real)
        ):
            raise ValidationError(
                "rank_tolerance must be a finite nonnegative "
                "real number."
            )

        tolerance_value = float(rank_tolerance)

        if (
            not np.isfinite(tolerance_value)
            or tolerance_value < 0.0
        ):
            raise ValidationError(
                "rank_tolerance must be a finite nonnegative "
                "real number."
            )

    rank_value: int | None = None

    if numerical_rank is not None:
        if (
            isinstance(numerical_rank, (bool, np.bool_))
            or not isinstance(numerical_rank, Integral)
        ):
            raise ValidationError(
                "numerical_rank must be an integer between zero "
                "and the number of singular values."
            )

        rank_value = int(numerical_rank)

        if rank_value < 0 or rank_value > values.size:
            raise ValidationError(
                "numerical_rank must be an integer between zero "
                "and the number of singular values."
            )

    sorted_values = np.sort(values.copy())[::-1]
    indices = np.arange(1, sorted_values.size + 1)
    figure, axis = _prepare_axes(ax)

    axis.plot(
        indices,
        sorted_values,
        marker="o",
        linestyle="-",
        label="Singular values",
    )

    if np.all(sorted_values > 0.0):
        axis.set_yscale("log")
    else:
        axis.set_yscale("linear")

    if tolerance_value is not None:
        axis.axhline(
            tolerance_value,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=f"Rank tolerance = {tolerance_value:.2e}",
        )

    if rank_value is not None:
        axis.axvline(
            rank_value + 0.5,
            color="tab:gray",
            linestyle=":",
            linewidth=1.2,
            label=f"Numerical rank = {rank_value}",
        )

    axis.set_xlim(0.5, sorted_values.size + 0.5)
    axis.set_xlabel("Singular-value index")
    axis.set_ylabel("Singular value")
    axis.set_title(title)
    axis.grid(True, which="both", linestyle=":", alpha=0.6)

    if tolerance_value is not None or rank_value is not None:
        axis.legend()

    return figure, axis
