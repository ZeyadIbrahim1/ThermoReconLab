"""Tests for ThermoReconLab visualizations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.sensors import SensorData
from thermoreconlab.visualization import (
    plot_error_map,
    plot_heatmap,
    plot_sensor_layout,
    plot_sensor_measurements,
    plot_source,
    plot_temperature,
)


@pytest.fixture(autouse=True)
def close_figures() -> None:
    """Close all figures after every test."""
    yield
    plt.close("all")


def test_plot_heatmap_returns_figure_and_axis() -> None:
    field = np.arange(20, dtype=float).reshape(4, 5)

    figure, axis = plot_heatmap(field, title="Example")

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)
    assert axis.get_title() == "Example"
    assert len(figure.axes) == 2


def test_plot_heatmap_uses_physical_grid() -> None:
    grid = Grid2D(nx=4, ny=5)
    field = np.ones(grid.shape)

    _, axis = plot_heatmap(field, grid=grid)

    assert axis.get_xlabel() == "x"
    assert axis.get_ylabel() == "y"


def test_plot_heatmap_rejects_wrong_shape() -> None:
    grid = Grid2D(nx=4, ny=5)

    with pytest.raises(ValidationError):
        plot_heatmap(
            np.ones((5, 4)),
            grid=grid,
        )


def test_plot_source_and_temperature_run() -> None:
    grid = Grid2D(nx=5, ny=6)
    field = np.ones(grid.shape)

    source_figure, source_axis = plot_source(
        field,
        grid=grid,
    )
    temperature_figure, temperature_axis = plot_temperature(
        field,
        grid=grid,
    )

    assert isinstance(source_figure, Figure)
    assert isinstance(temperature_figure, Figure)
    assert source_axis.get_title() == "Heat-source field"
    assert temperature_axis.get_title() == "Temperature field"


def test_error_map_uses_symmetric_limits() -> None:
    error = np.array(
        [
            [-2.0, 0.0],
            [1.0, 2.0],
        ]
    )

    _, axis = plot_error_map(error)

    image = axis.images[0]
    lower, upper = image.get_clim()

    assert lower == pytest.approx(-2.0)
    assert upper == pytest.approx(2.0)


def test_sensor_layout_returns_requested_points() -> None:
    grid = Grid2D(nx=6, ny=7)
    indices = np.array(
        [
            [1, 1],
            [2, 3],
            [4, 5],
        ]
    )

    _, axis = plot_sensor_layout(indices, grid)

    offsets = axis.collections[0].get_offsets()

    assert offsets.shape == (3, 2)
    assert np.allclose(
        offsets[:, 0],
        grid.x[indices[:, 0]],
    )
    assert np.allclose(
        offsets[:, 1],
        grid.y[indices[:, 1]],
    )


def test_sensor_layout_accepts_background() -> None:
    grid = Grid2D(nx=6, ny=7)
    indices = np.array([[1, 1], [3, 4]])
    background = np.ones(grid.shape)

    figure, axis = plot_sensor_layout(
        indices,
        grid,
        background=background,
    )

    assert isinstance(figure, Figure)
    assert len(axis.collections) >= 2
    assert len(figure.axes) == 2


def test_sensor_measurements_use_sensor_values() -> None:
    grid = Grid2D(nx=6, ny=7)
    sensor_data = SensorData(
        indices=np.array([[1, 1], [3, 4]]),
        values=np.array([0.2, 0.8]),
    )

    _, axis = plot_sensor_measurements(
        sensor_data,
        grid,
    )

    plotted_values = axis.collections[0].get_array()

    assert np.allclose(
        plotted_values,
        sensor_data.values,
    )


def test_sensor_layout_rejects_invalid_indices() -> None:
    grid = Grid2D(nx=6, ny=7)

    with pytest.raises(ValidationError):
        plot_sensor_layout(
            np.array([[1, 1], [6, 2]]),
            grid,
        )


def test_sensor_measurements_reject_invalid_object() -> None:
    grid = Grid2D(nx=6, ny=7)

    with pytest.raises(ValidationError):
        plot_sensor_measurements(
            np.ones(3),  # type: ignore[arg-type]
            grid,
        )


def test_plot_function_accepts_existing_axis() -> None:
    figure, axis = plt.subplots()
    field = np.ones((3, 4))

    returned_figure, returned_axis = plot_heatmap(
        field,
        ax=axis,
    )

    assert returned_figure is figure
    assert returned_axis is axis
