"""Scientific visualization tools for ThermoReconLab.

This module contains a small set of reusable Matplotlib functions for
two-dimensional fields, reconstruction errors, and sparse sensors.
"""
from __future__ import annotations

from typing import Final
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
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


def plot_heatmap(
    field: ArrayLike,
    *,
    grid: Grid2D | None = None,
    title: str = "",
    colorbar_label: str = "",
    ax: Axes | None = None,
    cmap: str = "viridis",
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

    figure, axis = _prepare_axes(ax)

    if grid is None:
        image = axis.imshow(
            field_array.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
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
) -> tuple[Figure, Axes]:
    """Plot a heat-source field."""
    return plot_heatmap(
        source,
        grid=grid,
        title=title,
        colorbar_label="Source intensity",
        ax=ax,
        cmap="inferno",
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