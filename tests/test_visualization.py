"""Tests for ThermoReconLab visualizations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.sensors import SensorData
from thermoreconlab.visualization import (
    plot_error_map,
    plot_heatmap,
    plot_noise_sensitivity_study,
    plot_reconstruction_comparison,
    plot_reconstruction_method_comparison,
    plot_regularization_study,
    plot_repeated_noise_study,
    plot_sensor_count_study,
    plot_sensor_layout,
    plot_sensor_layout_study,
    plot_sensor_measurements,
    plot_singular_values,
    plot_source,
    plot_temperature,
)


@pytest.fixture(autouse=True)
def close_figures() -> None:  # type: ignore
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


def test_regularization_plot_returns_log_axis() -> None:
    """Regularization results should be plotted on a log alpha axis."""
    dataframe = pd.DataFrame(
        {
            "alpha": [1e-3, 1e-4, 1e-5],
            "relative_l2_error": [0.9, 0.7, 0.6],
        }
    )

    figure, axis = plot_regularization_study(dataframe)

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)
    assert axis.get_xscale() == "log"
    assert len(axis.lines) == 1


def test_regularization_plot_sorts_alpha_values() -> None:
    """Alpha values should appear in ascending numerical order."""
    dataframe = pd.DataFrame(
        {
            "alpha": [1e-3, 1e-5, 1e-4],
            "relative_l2_error": [0.9, 0.6, 0.7],
        }
    )

    _, axis = plot_regularization_study(dataframe)

    plotted_alpha = axis.lines[0].get_xdata()

    assert np.array_equal(
        plotted_alpha,
        np.array([1e-5, 1e-4, 1e-3]),
    )


def test_regularization_plot_rejects_missing_metric() -> None:
    """The selected metric must exist in the DataFrame."""
    dataframe = pd.DataFrame(
        {
            "alpha": [1e-3, 1e-4],
            "rmse": [0.2, 0.1],
        }
    )

    with pytest.raises(ValidationError):
        plot_regularization_study(
            dataframe,
            metric="relative_l2_error",
        )


def test_regularization_plot_rejects_nonpositive_alpha() -> None:
    """Alpha values must remain positive."""
    dataframe = pd.DataFrame(
        {
            "alpha": [1e-3, 0.0],
            "relative_l2_error": [0.8, 0.6],
        }
    )

    with pytest.raises(ValidationError):
        plot_regularization_study(dataframe)


def test_sensor_count_plot_returns_figure_and_axis() -> None:
    """Sensor-count results should produce a line plot."""
    dataframe = pd.DataFrame(
        {
            "sensor_count": [4, 9, 16],
            "relative_l2_error": [0.9, 0.7, 0.5],
        }
    )

    figure, axis = plot_sensor_count_study(dataframe)

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)
    assert len(axis.lines) == 1
    assert axis.get_xlabel() == "Number of sensors"


def test_sensor_count_plot_sorts_counts() -> None:
    """Sensor counts should be plotted in ascending order."""
    dataframe = pd.DataFrame(
        {
            "sensor_count": [16, 4, 9],
            "relative_l2_error": [0.5, 0.9, 0.7],
        }
    )

    _, axis = plot_sensor_count_study(dataframe)

    plotted_counts = axis.lines[0].get_xdata()

    assert np.array_equal(
        plotted_counts,
        np.array([4.0, 9.0, 16.0]),
    )


def test_sensor_count_plot_rejects_missing_metric() -> None:
    """The selected metric must exist in the DataFrame."""
    dataframe = pd.DataFrame(
        {
            "sensor_count": [4, 9],
            "rmse": [0.2, 0.1],
        }
    )

    with pytest.raises(ValidationError):
        plot_sensor_count_study(
            dataframe,
            metric="relative_l2_error",
        )


def test_sensor_count_plot_rejects_nonpositive_count() -> None:
    """Sensor counts must be positive."""
    dataframe = pd.DataFrame(
        {
            "sensor_count": [0, 9],
            "relative_l2_error": [0.9, 0.7],
        }
    )

    with pytest.raises(ValidationError):
        plot_sensor_count_study(dataframe)        


def test_noise_study_plot_returns_figure_and_axis() -> None:
    """Noise-study results should produce a line plot."""
    dataframe = pd.DataFrame(
        {
            "noise_level": [0.0, 0.02, 0.05],
            "relative_l2_error": [0.50, 0.53, 0.64],
        }
    )

    figure, axis = plot_noise_sensitivity_study(dataframe)

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)
    assert len(axis.lines) == 1
    assert (
        axis.get_xlabel()
        == "Relative measurement-noise level"
    )


def test_noise_study_plot_sorts_noise_levels() -> None:
    """Noise levels should be plotted in ascending order."""
    dataframe = pd.DataFrame(
        {
            "noise_level": [0.05, 0.0, 0.02],
            "relative_l2_error": [0.64, 0.50, 0.53],
        }
    )

    _, axis = plot_noise_sensitivity_study(dataframe)

    plotted_levels = axis.lines[0].get_xdata()

    assert np.array_equal(
        plotted_levels,
        np.array([0.0, 0.02, 0.05]),
    )


def test_noise_study_plot_rejects_missing_metric() -> None:
    """The selected metric must exist in the DataFrame."""
    dataframe = pd.DataFrame(
        {
            "noise_level": [0.0, 0.02],
            "rmse": [0.08, 0.09],
        }
    )

    with pytest.raises(ValidationError):
        plot_noise_sensitivity_study(
            dataframe,
            metric="relative_l2_error",
        )


def test_noise_study_plot_rejects_negative_noise() -> None:
    """Noise levels cannot be negative."""
    dataframe = pd.DataFrame(
        {
            "noise_level": [-0.01, 0.02],
            "relative_l2_error": [0.50, 0.53],
        }
    )

    with pytest.raises(ValidationError):
        plot_noise_sensitivity_study(dataframe)


def repeated_noise_summary() -> pd.DataFrame:
    """Return a small unsorted repeated-noise summary."""
    return pd.DataFrame(
        {
            "noise_level": [0.05, 0.0, 0.02],
            "mean_relative_l2_error": [0.64, 0.50, 0.55],
            "std_relative_l2_error": [0.04, 0.00, 0.02],
        }
    )


def test_repeated_noise_plot_returns_figure_and_axis() -> None:
    figure, axis = plot_repeated_noise_study(
        repeated_noise_summary()
    )

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)


def test_repeated_noise_plot_sorts_noise_levels() -> None:
    _, axis = plot_repeated_noise_study(
        repeated_noise_summary()
    )

    plotted_levels = axis.lines[0].get_xdata()

    assert np.array_equal(
        plotted_levels,
        np.array([0.0, 0.02, 0.05]),
    )


def test_repeated_noise_plot_creates_error_bars() -> None:
    _, axis = plot_repeated_noise_study(
        repeated_noise_summary()
    )

    assert len(axis.collections) >= 1


def test_repeated_noise_plot_has_required_labels() -> None:
    _, axis = plot_repeated_noise_study(
        repeated_noise_summary()
    )

    assert axis.get_xlabel() == "Relative measurement-noise level"
    assert axis.get_ylabel() == "Mean relative L2 source error"
    assert axis.get_title() == "Repeated-noise study"


def test_repeated_noise_plot_rejects_missing_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "noise_level": [0.0, 0.02],
            "mean_relative_l2_error": [0.50, 0.55],
        }
    )

    with pytest.raises(ValidationError):
        plot_repeated_noise_study(dataframe)


def test_repeated_noise_plot_rejects_empty_input() -> None:
    with pytest.raises(ValidationError):
        plot_repeated_noise_study(pd.DataFrame())


def test_repeated_noise_plot_rejects_negative_std() -> None:
    dataframe = repeated_noise_summary()
    dataframe.loc[0, "std_relative_l2_error"] = -0.01

    with pytest.raises(ValidationError):
        plot_repeated_noise_study(dataframe)


def test_repeated_noise_plot_rejects_non_finite_values() -> None:
    dataframe = repeated_noise_summary()
    dataframe.loc[0, "mean_relative_l2_error"] = np.nan

    with pytest.raises(ValidationError):
        plot_repeated_noise_study(dataframe)


def test_repeated_noise_plot_rejects_duplicate_noise_levels() -> None:
    dataframe = repeated_noise_summary()
    dataframe.loc[0, "noise_level"] = 0.02

    with pytest.raises(ValidationError):
        plot_repeated_noise_study(dataframe)


def test_repeated_noise_plot_does_not_modify_input() -> None:
    dataframe = repeated_noise_summary()
    original = dataframe.copy(deep=True)

    plot_repeated_noise_study(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_repeated_noise_plot_saves_with_agg_backend(
    tmp_path,
) -> None:
    figure, _ = plot_repeated_noise_study(
        repeated_noise_summary()
    )
    output_path = tmp_path / "repeated_noise.png"

    figure.savefig(output_path, dpi=100, bbox_inches="tight")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def sensor_layout_summary() -> pd.DataFrame:
    """Return a small sensor-layout summary in a fixed order."""
    return pd.DataFrame(
        {
            "strategy": [
                "center_focused",
                "regular",
                "random",
            ],
            "mean_relative_l2_error": [0.48, 0.52, 0.57],
            "std_relative_l2_error": [0.00, 0.02, 0.05],
        }
    )


def test_sensor_layout_study_plot_returns_figure_and_axis() -> None:
    figure, axis = plot_sensor_layout_study(
        sensor_layout_summary()
    )

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)


def test_sensor_layout_study_plot_displays_each_strategy() -> None:
    _, axis = plot_sensor_layout_study(
        sensor_layout_summary()
    )

    assert len(axis.patches) == 3
    assert [
        tick.get_text()
        for tick in axis.get_xticklabels()
    ] == [
        "Center Focused",
        "Regular",
        "Random",
    ]


def test_sensor_layout_study_plot_creates_error_bars() -> None:
    _, axis = plot_sensor_layout_study(
        sensor_layout_summary()
    )

    assert len(axis.collections) >= 1


def test_sensor_layout_study_plot_has_required_labels() -> None:
    _, axis = plot_sensor_layout_study(
        sensor_layout_summary()
    )

    assert axis.get_xlabel() == "Sensor strategy"
    assert axis.get_ylabel() == "Mean relative L2 source error"
    assert axis.get_title() == "Sensor-layout comparison"


def test_sensor_layout_study_plot_rejects_missing_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "strategy": ["regular"],
            "mean_relative_l2_error": [0.5],
        }
    )

    with pytest.raises(ValidationError):
        plot_sensor_layout_study(dataframe)


def test_sensor_layout_study_plot_rejects_empty_input() -> None:
    with pytest.raises(ValidationError):
        plot_sensor_layout_study(pd.DataFrame())


def test_sensor_layout_study_plot_rejects_negative_std() -> None:
    dataframe = sensor_layout_summary()
    dataframe.loc[0, "std_relative_l2_error"] = -0.01

    with pytest.raises(ValidationError):
        plot_sensor_layout_study(dataframe)


def test_sensor_layout_study_plot_rejects_duplicate_strategy() -> None:
    dataframe = sensor_layout_summary()
    dataframe.loc[0, "strategy"] = "regular"

    with pytest.raises(ValidationError):
        plot_sensor_layout_study(dataframe)


def test_sensor_layout_study_plot_does_not_modify_input() -> None:
    dataframe = sensor_layout_summary()
    original = dataframe.copy(deep=True)

    plot_sensor_layout_study(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_sensor_layout_study_plot_saves_successfully(
    tmp_path,
) -> None:
    figure, _ = plot_sensor_layout_study(
        sensor_layout_summary()
    )
    output_path = tmp_path / "sensor_layout_study.png"

    figure.savefig(output_path, dpi=100, bbox_inches="tight")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def create_comparison_inputs():
    """Create deterministic fields for comparison-plot tests."""
    grid = Grid2D(nx=6, ny=7)

    true_source = np.zeros(grid.shape)
    true_source[2:4, 2:5] = 1.0

    reconstructed_source = np.zeros(grid.shape)
    reconstructed_source[2:5, 2:5] = 0.6

    temperature = np.arange(
        grid.size,
        dtype=float,
    ).reshape(grid.shape)

    sensor_data = SensorData(
        indices=np.array(
            [
                [1, 1],
                [3, 3],
                [4, 5],
            ]
        ),
        values=np.array([0.1, 0.2, 0.3]),
    )

    return (
        grid,
        true_source,
        temperature,
        sensor_data,
        reconstructed_source,
    )


def test_heatmap_accepts_explicit_color_limits() -> None:
    field = np.arange(12, dtype=float).reshape(3, 4)

    _, axis = plot_heatmap(
        field,
        vmin=-1.0,
        vmax=20.0,
    )

    assert axis.images[0].get_clim() == (-1.0, 20.0)


@pytest.mark.parametrize(
    ("vmin", "vmax"),
    [
        (1.0, 1.0),
        (2.0, 1.0),
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (True, 1.0),
    ],
)
def test_heatmap_rejects_invalid_color_limits(
    vmin: object,
    vmax: object,
) -> None:
    with pytest.raises(ValidationError):
        plot_heatmap(
            np.ones((3, 4)),
            vmin=vmin,  # type: ignore[arg-type]
            vmax=vmax,  # type: ignore[arg-type]
        )


def test_source_accepts_explicit_color_limits() -> None:
    _, axis = plot_source(
        np.ones((3, 4)),
        vmin=0.0,
        vmax=2.0,
    )

    assert axis.images[0].get_clim() == (0.0, 2.0)


def test_reconstruction_comparison_returns_expected_axes() -> None:
    figure, axes = plot_reconstruction_comparison(
        *create_comparison_inputs()
    )

    assert isinstance(figure, Figure)
    assert axes.shape == (2, 2)
    assert all(
        isinstance(axis, Axes)
        for axis in axes.ravel()
    )


def test_comparison_uses_common_source_limits() -> None:
    _, axes = plot_reconstruction_comparison(
        *create_comparison_inputs()
    )

    true_limits = axes[0, 0].collections[0].get_clim()
    reconstructed_limits = axes[1, 0].collections[0].get_clim()

    assert true_limits == reconstructed_limits


def test_comparison_error_limits_are_symmetric() -> None:
    _, axes = plot_reconstruction_comparison(
        *create_comparison_inputs()
    )

    lower, upper = axes[1, 1].collections[0].get_clim()

    assert lower == pytest.approx(-upper)


def test_comparison_displays_sensor_markers() -> None:
    _, axes = plot_reconstruction_comparison(
        *create_comparison_inputs()
    )

    scatter_objects = [
        collection
        for collection in axes[0, 1].collections
        if isinstance(collection, PathCollection)
    ]

    assert len(scatter_objects) == 1
    assert scatter_objects[0].get_offsets().shape == (3, 2)


def test_comparison_accepts_explicit_source_limits() -> None:
    _, axes = plot_reconstruction_comparison(
        *create_comparison_inputs(),
        source_vmin=-0.5,
        source_vmax=1.5,
    )

    assert axes[0, 0].collections[0].get_clim() == (-0.5, 1.5)
    assert axes[1, 0].collections[0].get_clim() == (-0.5, 1.5)


def test_comparison_rejects_invalid_source_limits() -> None:
    with pytest.raises(ValidationError):
        plot_reconstruction_comparison(
            *create_comparison_inputs(),
            source_vmin=1.0,
            source_vmax=0.0,
        )


def test_comparison_rejects_mismatched_shapes() -> None:
    (
        grid,
        true_source,
        temperature,
        sensor_data,
        _,
    ) = create_comparison_inputs()

    with pytest.raises(ValidationError):
        plot_reconstruction_comparison(
            grid,
            true_source,
            temperature,
            sensor_data,
            np.ones((5, 5)),
        )


def test_comparison_does_not_modify_inputs() -> None:
    inputs = create_comparison_inputs()

    true_copy = inputs[1].copy()
    temperature_copy = inputs[2].copy()
    sensor_indices_copy = inputs[3].indices.copy()
    sensor_values_copy = inputs[3].values.copy()
    reconstructed_copy = inputs[4].copy()

    plot_reconstruction_comparison(*inputs)

    assert np.array_equal(inputs[1], true_copy)
    assert np.array_equal(inputs[2], temperature_copy)
    assert np.array_equal(
        inputs[3].indices,
        sensor_indices_copy,
    )
    assert np.array_equal(
        inputs[3].values,
        sensor_values_copy,
    )
    assert np.array_equal(inputs[4], reconstructed_copy)


def test_comparison_figure_saves_successfully(tmp_path) -> None:
    figure, _ = plot_reconstruction_comparison(
        *create_comparison_inputs()
    )
    output_path = tmp_path / "comparison.png"

    figure.savefig(output_path, dpi=100, bbox_inches="tight")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def create_method_comparison_inputs():
    """Create deterministic fields for method-comparison tests."""
    (
        grid,
        true_source,
        temperature,
        sensor_data,
        identity_source,
    ) = create_comparison_inputs()
    smooth_source = np.maximum(identity_source, 0.0) * 0.9
    compact_source = np.zeros(grid.shape)
    compact_source[2:4, 2:4] = 0.8

    return (
        grid,
        true_source,
        temperature,
        sensor_data,
        identity_source,
        smooth_source,
        compact_source,
    )


def test_method_comparison_returns_expected_axes() -> None:
    figure, axes = plot_reconstruction_method_comparison(
        *create_method_comparison_inputs()
    )

    assert isinstance(figure, Figure)
    assert axes.shape == (2, 4)
    assert all(isinstance(axis, Axes) for axis in axes.ravel())
    assert len(figure.axes) == 11


def test_method_comparison_has_eight_panel_titles() -> None:
    _, axes = plot_reconstruction_method_comparison(
        *create_method_comparison_inputs()
    )

    assert [axis.get_title() for axis in axes.ravel()] == [
        "(a) True source",
        "(b) Identity Tikhonov",
        "(c) Smooth nonnegative",
        "(d) Compact nonnegative",
        "(e) Temperature and sensors",
        "(f) Identity signed error",
        "(g) Smooth signed error",
        "(h) Compact signed error",
    ]


def test_method_comparison_uses_common_source_limits() -> None:
    _, axes = plot_reconstruction_method_comparison(
        *create_method_comparison_inputs()
    )
    limits = [
        axis.collections[0].get_clim()
        for axis in axes[0]
    ]

    assert all(limit == limits[0] for limit in limits[1:])


def test_method_comparison_uses_common_symmetric_error_limits() -> None:
    _, axes = plot_reconstruction_method_comparison(
        *create_method_comparison_inputs()
    )
    limits = [
        axis.collections[0].get_clim()
        for axis in axes[1, 1:]
    ]

    assert all(limit == limits[0] for limit in limits[1:])
    assert limits[0][0] == pytest.approx(-limits[0][1])


def test_method_comparison_accepts_explicit_limits() -> None:
    _, axes = plot_reconstruction_method_comparison(
        *create_method_comparison_inputs(),
        source_vmin=-0.5,
        source_vmax=1.5,
        error_limit=2.0,
    )

    assert all(
        axis.collections[0].get_clim() == (-0.5, 1.5)
        for axis in axes[0]
    )
    assert all(
        axis.collections[0].get_clim() == (-2.0, 2.0)
        for axis in axes[1, 1:]
    )


def test_method_comparison_displays_all_sensor_markers() -> None:
    inputs = create_method_comparison_inputs()
    _, axes = plot_reconstruction_method_comparison(*inputs)
    scatter_objects = [
        collection
        for collection in axes[1, 0].collections
        if isinstance(collection, PathCollection)
    ]

    assert len(scatter_objects) == 1
    assert scatter_objects[0].get_offsets().shape == (
        len(inputs[3]),
        2,
    )


def test_method_comparison_uses_physical_axis_labels() -> None:
    _, axes = plot_reconstruction_method_comparison(
        *create_method_comparison_inputs()
    )

    assert all(axis.get_xlabel() == "x" for axis in axes.ravel())
    assert all(axis.get_ylabel() == "y" for axis in axes.ravel())


@pytest.mark.parametrize("field_position", [1, 2, 4, 5, 6])
def test_method_comparison_rejects_invalid_field_shapes(
    field_position: int,
) -> None:
    inputs = list(create_method_comparison_inputs())
    inputs[field_position] = np.ones((5, 5))

    with pytest.raises(ValidationError):
        plot_reconstruction_method_comparison(*inputs)


@pytest.mark.parametrize(
    ("source_vmin", "source_vmax"),
    [
        (1.0, 1.0),
        (2.0, 1.0),
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (True, 1.0),
    ],
)
def test_method_comparison_rejects_invalid_source_limits(
    source_vmin: object,
    source_vmax: object,
) -> None:
    with pytest.raises(ValidationError):
        plot_reconstruction_method_comparison(
            *create_method_comparison_inputs(),
            source_vmin=source_vmin,  # type: ignore[arg-type]
            source_vmax=source_vmax,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_limit",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_method_comparison_rejects_invalid_error_limit(
    invalid_limit: object,
) -> None:
    with pytest.raises(ValidationError):
        plot_reconstruction_method_comparison(
            *create_method_comparison_inputs(),
            error_limit=invalid_limit,  # type: ignore[arg-type]
        )


def test_method_comparison_rejects_nonfinite_field() -> None:
    inputs = list(create_method_comparison_inputs())
    invalid_source = inputs[5].copy()
    invalid_source[2, 2] = np.nan
    inputs[5] = invalid_source

    with pytest.raises(ValidationError):
        plot_reconstruction_method_comparison(*inputs)


def test_method_comparison_handles_constant_fields() -> None:
    inputs = list(create_method_comparison_inputs())

    for position in (1, 2, 4, 5, 6):
        inputs[position] = np.ones(inputs[0].shape)

    _, axes = plot_reconstruction_method_comparison(*inputs)
    lower, upper = axes[0, 0].collections[0].get_clim()

    assert np.isfinite([lower, upper]).all()
    assert lower < upper


def test_method_comparison_handles_zero_error() -> None:
    inputs = list(create_method_comparison_inputs())
    inputs[4] = inputs[1].copy()
    inputs[5] = inputs[1].copy()
    inputs[6] = inputs[1].copy()

    _, axes = plot_reconstruction_method_comparison(*inputs)

    assert all(
        axis.collections[0].get_clim() == (-1.0, 1.0)
        for axis in axes[1, 1:]
    )


def test_method_comparison_does_not_modify_inputs() -> None:
    inputs = create_method_comparison_inputs()
    array_copies = {
        position: inputs[position].copy()
        for position in (1, 2, 4, 5, 6)
    }
    sensor_indices = inputs[3].indices.copy()
    sensor_values = inputs[3].values.copy()

    plot_reconstruction_method_comparison(*inputs)

    for position, original in array_copies.items():
        assert np.array_equal(inputs[position], original)

    assert np.array_equal(inputs[3].indices, sensor_indices)
    assert np.array_equal(inputs[3].values, sensor_values)


def test_method_comparison_figure_saves_successfully(tmp_path) -> None:
    figure, _ = plot_reconstruction_method_comparison(
        *create_method_comparison_inputs()
    )
    output_path = tmp_path / "method-comparison.png"

    figure.savefig(output_path, dpi=100, bbox_inches="tight")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_singular_value_plot_returns_figure_and_axis() -> None:
    figure, axis = plot_singular_values([3.0, 2.0, 1.0])

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)


def test_singular_value_plot_sorts_values_descending() -> None:
    _, axis = plot_singular_values([1.0, 3.0, 2.0])

    assert np.array_equal(axis.lines[0].get_xdata(), [1, 2, 3])
    assert np.array_equal(
        axis.lines[0].get_ydata(),
        [3.0, 2.0, 1.0],
    )


def test_singular_value_plot_uses_log_scale_for_positive_values() -> None:
    _, axis = plot_singular_values([3.0, 2.0, 1.0])

    assert axis.get_yscale() == "log"


def test_singular_value_plot_uses_linear_scale_and_keeps_zeros() -> None:
    _, axis = plot_singular_values([3.0, 0.0, 1.0])

    assert axis.get_yscale() == "linear"
    assert np.array_equal(
        axis.lines[0].get_ydata(),
        [3.0, 1.0, 0.0],
    )


def test_singular_value_plot_shows_rank_tolerance() -> None:
    _, axis = plot_singular_values(
        [3.0, 2.0, 1.0],
        rank_tolerance=0.5,
    )

    tolerance_lines = [
        line
        for line in axis.lines
        if line.get_label().startswith("Rank tolerance")
    ]
    assert len(tolerance_lines) == 1
    assert np.allclose(tolerance_lines[0].get_ydata(), 0.5)


def test_singular_value_plot_shows_numerical_rank() -> None:
    _, axis = plot_singular_values(
        [3.0, 2.0, 1.0],
        numerical_rank=2,
    )

    rank_lines = [
        line
        for line in axis.lines
        if line.get_label() == "Numerical rank = 2"
    ]
    assert len(rank_lines) == 1
    assert np.allclose(rank_lines[0].get_xdata(), 2.5)


@pytest.mark.parametrize(
    "invalid_values",
    [
        [],
        [[1.0, 2.0]],
        [1.0, -1.0],
        [1.0, float("nan")],
        [1.0, float("inf")],
        ["invalid"],
    ],
)
def test_singular_value_plot_rejects_invalid_values(
    invalid_values: object,
) -> None:
    with pytest.raises(ValidationError):
        plot_singular_values(invalid_values)


@pytest.mark.parametrize(
    "invalid_tolerance",
    [-1.0, float("nan"), float("inf"), True, "invalid"],
)
def test_singular_value_plot_rejects_invalid_tolerance(
    invalid_tolerance: object,
) -> None:
    with pytest.raises(ValidationError):
        plot_singular_values(
            [3.0, 2.0, 1.0],
            rank_tolerance=invalid_tolerance,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_rank",
    [-1, 4, 1.5, True],
)
def test_singular_value_plot_rejects_invalid_rank(
    invalid_rank: object,
) -> None:
    with pytest.raises(ValidationError):
        plot_singular_values(
            [3.0, 2.0, 1.0],
            numerical_rank=invalid_rank,  # type: ignore[arg-type]
        )


def test_singular_value_plot_does_not_modify_input() -> None:
    values = np.array([1.0, 3.0, 2.0])
    original = values.copy()

    plot_singular_values(values)

    assert np.array_equal(values, original)


def test_singular_value_plot_saves_successfully(tmp_path) -> None:
    figure, _ = plot_singular_values(
        [3.0, 2.0, 1.0],
        rank_tolerance=0.5,
        numerical_rank=3,
    )
    output_path = tmp_path / "singular-values.png"

    figure.savefig(output_path, dpi=100, bbox_inches="tight")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
