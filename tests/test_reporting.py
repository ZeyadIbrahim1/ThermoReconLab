"""Tests for result reporting and export."""

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pandas.testing import assert_frame_equal

from thermoreconlab import (
    reconstruct_from_measurements,
    reconstruct_from_temperature_field,
    run_synthetic_benchmark,
)
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.data import gaussian_source
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.reconstruction import solve_forward
from thermoreconlab.reporting import (
    export_results,
    save_config_json,
    save_figures,
    save_metrics_csv,
    write_markdown_report,
)
import matplotlib.pyplot as plt
from thermoreconlab.sensors import (
    create_sensor_data,
    regular_grid_sensors,
)
from thermoreconlab.visualization import plot_source


SOURCE_METRIC_NAMES = {
    "rmse",
    "mae",
    "relative_l2_error",
    "max_absolute_error",
}
MEASUREMENT_METRIC_NAMES = {
    "residual_norm",
    "relative_residual",
    "residual_rms",
    "solution_norm",
}


def create_synthetic_result():
    """Create a small deterministic synthetic benchmark."""
    return run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        alpha=1e-4,
        regularization="identity",
        seed=42,
    )


def create_user_result():
    """Create a small user-measurement reconstruction."""
    grid = Grid2D(nx=8, ny=8)
    source = gaussian_source(grid, sigma=0.12)
    temperature = solve_forward(source, grid)
    indices = regular_grid_sensors(grid, count=9)
    sensor_data = create_sensor_data(
        temperature,
        indices,
        grid,
    )

    return reconstruct_from_measurements(
        sensor_data,
        grid_shape=grid.shape,
        alpha=1e-4,
    )


def create_temperature_result():
    """Create a small temperature-field reconstruction."""
    grid = Grid2D(nx=8, ny=8)
    source = gaussian_source(grid, sigma=0.12)
    temperature = solve_forward(source, grid)
    return reconstruct_from_temperature_field(
        temperature,
        grid=grid,
        alpha=1e-4,
    )


def remove_runtime_values(value):
    """Recursively remove only runtime-named fields."""
    if isinstance(value, dict):
        return {
            key: remove_runtime_values(item)
            for key, item in value.items()
            if "runtime" not in key.lower()
        }
    if isinstance(value, list):
        return [remove_runtime_values(item) for item in value]
    return value


def test_save_config_json_creates_file(tmp_path: Path) -> None:
    """Synthetic JSON should preserve and extend the result structure."""
    result = create_synthetic_result()

    path = save_config_json(result, tmp_path)
    content = json.loads(path.read_text(encoding="utf-8"))

    assert path.exists()
    assert path.name == "summary.json"
    assert {"config", "metrics", "runtime", "reconstruction"} <= set(
        content
    )
    assert "reporting_summary" in content
    summary = content["reporting_summary"]
    assert summary["num_unknowns"] == 36
    assert summary["num_measurements"] == 9
    assert summary["measurement_to_unknown_ratio"] == pytest.approx(
        9 / 36
    )
    assert summary["regularization"] == "identity"
    assert {
        "numerical_rank",
        "nullity",
        "is_underdetermined",
        "rank_tolerance",
    } <= set(summary)


def test_save_metrics_csv_creates_expected_columns(
    tmp_path: Path,
) -> None:
    """Synthetic CSV should contain source and measurement metrics."""
    result = create_synthetic_result()

    path = save_metrics_csv(result, tmp_path)
    dataframe = pd.read_csv(path)

    assert path.exists()
    assert len(dataframe) == 1
    required_columns = {
        *SOURCE_METRIC_NAMES,
        *MEASUREMENT_METRIC_NAMES,
        "regularization",
        "alpha",
        "num_measurements",
        "num_unknowns",
        "measurement_to_unknown_ratio",
        "numerical_rank",
        "nullity",
        "is_underdetermined",
    }
    assert required_columns <= set(dataframe.columns)
    assert dataframe.loc[0, "mode"] == "synthetic_benchmark"
    assert dataframe.loc[0, "num_unknowns"] == 36
    assert dataframe.loc[0, "num_measurements"] == 9
    assert dataframe.loc[
        0, "measurement_to_unknown_ratio"
    ] == pytest.approx(9 / 36)


def test_user_metrics_csv_omits_ground_truth_metrics(
    tmp_path: Path,
) -> None:
    """User results should not claim unavailable source errors."""
    result = create_user_result()

    path = save_metrics_csv(result, tmp_path)
    dataframe = pd.read_csv(path)

    assert MEASUREMENT_METRIC_NAMES <= set(dataframe.columns)
    assert SOURCE_METRIC_NAMES.isdisjoint(dataframe.columns)
    assert dataframe.loc[0, "regularization"] == "identity"
    assert dataframe.loc[0, "mode"] == "user_measurements"


def test_temperature_metrics_are_measurement_only(tmp_path: Path) -> None:
    """Temperature-field CSV should identify its mode without source errors."""
    path = save_metrics_csv(create_temperature_result(), tmp_path)
    dataframe = pd.read_csv(path)

    assert dataframe.loc[0, "mode"] == "temperature_field"
    assert dataframe.loc[0, "measurement_selection"] == "all_interior"
    assert MEASUREMENT_METRIC_NAMES <= set(dataframe.columns)
    assert SOURCE_METRIC_NAMES.isdisjoint(dataframe.columns)


def test_user_summary_has_measurement_metrics_only(tmp_path: Path) -> None:
    """User JSON should add diagnostics without source-error metrics."""
    path = save_config_json(create_user_result(), tmp_path)
    content = json.loads(path.read_text(encoding="utf-8"))

    assert MEASUREMENT_METRIC_NAMES <= set(content["metrics"])
    assert SOURCE_METRIC_NAMES.isdisjoint(content["metrics"])
    assert SOURCE_METRIC_NAMES.isdisjoint(content["reporting_summary"])


def test_save_figures_creates_synthetic_figure_set(
    tmp_path: Path,
) -> None:
    """Synthetic export should create six standard figures."""
    result = create_synthetic_result()

    paths = save_figures(result, tmp_path, dpi=80)

    assert set(paths) == {
        "reconstructed_source",
        "sensor_measurements",
        "true_source",
        "temperature",
        "error_map",
        "reconstruction_comparison",
    }
    assert all(
        path.exists() and path.stat().st_size > 0
        for path in paths.values()
    )
    comparison_path = paths["reconstruction_comparison"]
    assert plt.imread(comparison_path).size > 0


def test_user_figure_export_contains_no_ground_truth(
    tmp_path: Path,
) -> None:
    """User mode should export only available information."""
    result = create_user_result()

    paths = save_figures(result, tmp_path, dpi=80)

    assert set(paths) == {
        "reconstructed_source",
        "sensor_measurements",
    }


def test_temperature_figures_contain_no_ground_truth(
    tmp_path: Path,
) -> None:
    """Temperature-field mode should export only retained information."""
    paths = save_figures(create_temperature_result(), tmp_path, dpi=80)

    assert set(paths) == {
        "reconstructed_source",
        "sensor_measurements",
    }


def test_markdown_report_creates_file(
    tmp_path: Path,
) -> None:
    """Synthetic Markdown should contain validation and limitations."""
    result = create_synthetic_result()

    path = write_markdown_report(result, tmp_path)

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "# ThermoReconLab Result Report" in content
    assert "Ground-truth validation" in content
    assert (
        "Source-error metrics are evaluated on interior source nodes."
        in content
    )
    for label in (
        "RMSE",
        "MAE",
        "Relative L2 Error",
        "Maximum Absolute Error",
        "Regularization",
        "Interior source unknowns",
        "Measurement-to-unknown ratio",
        "Observation-matrix diagnostics",
        "Negative reconstruction artifacts",
        "sensor geometry",
        "selected alpha",
    ):
        assert label in content


def test_user_markdown_has_no_source_metrics(tmp_path: Path) -> None:
    """User Markdown should make the absence of source truth explicit."""
    path = write_markdown_report(create_user_result(), tmp_path)
    content = path.read_text(encoding="utf-8")
    required_sentence = (
        "Ground-truth source metrics are unavailable for this workflow."
    )

    assert content.count(required_sentence) == 1
    for forbidden_text in (
        "RMSE",
        "MAE",
        "Relative L2 Error",
        "Maximum Absolute Error",
        "Ground-truth validation",
    ):
        assert forbidden_text not in content
    assert "Measurement-space diagnostics" in content
    assert "Residual agreement does not prove source accuracy" in content


def test_temperature_markdown_has_no_source_metrics(tmp_path: Path) -> None:
    """Temperature-field Markdown should follow measurement-mode reporting."""
    path = write_markdown_report(create_temperature_result(), tmp_path)
    content = path.read_text(encoding="utf-8")

    assert (
        "Ground-truth source metrics are unavailable for this workflow."
        in content
    )
    assert "Ground-truth validation" not in content
    assert "Relative L2 Error" not in content


def test_export_results_creates_complete_package(
    tmp_path: Path,
) -> None:
    """Full export should create tables, report, and figures."""
    result = create_synthetic_result()

    exported = export_results(result, tmp_path, dpi=80)

    assert set(exported) == {
        "metrics",
        "summary",
        "report",
        "figures",
        "output_dir",
    }
    assert exported["metrics"].exists()
    assert exported["summary"].exists()
    assert exported["report"].exists()
    assert exported["output_dir"] == tmp_path
    assert all(
        path.exists()
        for path in exported["figures"].values()
    )
    assert "reconstruction_comparison" in exported["figures"]
    report_content = exported["report"].read_text(encoding="utf-8")
    assert (
        "figures/reconstruction_comparison.png"
        in report_content
    )


def test_reporting_uses_agg_backend() -> None:
    """Reporting should retain the noninteractive backend."""
    assert matplotlib.get_backend().lower() == "agg"


def test_save_figures_closes_every_figure(tmp_path: Path) -> None:
    """Standard figure export should leave no figures open."""
    plt.close("all")

    save_figures(create_synthetic_result(), tmp_path, dpi=80)

    assert plt.get_fignums() == []


def test_save_figures_closes_figure_when_saving_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed save should still close the figure it was given."""
    plt.close("all")

    def fail_savefig(self, path, **kwargs):
        raise OSError("simulated save failure")

    monkeypatch.setattr(Figure, "savefig", fail_savefig)

    with pytest.raises(OSError, match="simulated save failure"):
        save_figures(create_user_result(), tmp_path, dpi=80)

    assert plt.get_fignums() == []


def test_save_figures_uses_default_300_dpi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default raster export resolution should be 300 DPI."""
    recorded_dpi: list[int] = []

    def record_savefig(self, path, **kwargs):
        recorded_dpi.append(kwargs["dpi"])

    monkeypatch.setattr(Figure, "savefig", record_savefig)

    save_figures(create_user_result(), tmp_path)

    assert recorded_dpi == [300, 300]


def test_export_results_uses_default_300_dpi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete exports should forward their 300-DPI default."""
    recorded_dpi: list[int] = []

    def record_savefig(self, path, **kwargs):
        recorded_dpi.append(kwargs["dpi"])

    monkeypatch.setattr(Figure, "savefig", record_savefig)

    export_results(create_user_result(), tmp_path)

    assert recorded_dpi == [300, 300]


def test_save_figures_respects_explicit_dpi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly supplied raster resolution must be unchanged."""
    recorded_dpi: list[int] = []

    def record_savefig(self, path, **kwargs):
        recorded_dpi.append(kwargs["dpi"])

    monkeypatch.setattr(Figure, "savefig", record_savefig)

    save_figures(create_user_result(), tmp_path, dpi=123)

    assert recorded_dpi == [123, 123]


def test_pdf_figure_saving_remains_supported(tmp_path: Path) -> None:
    """The unchanged Matplotlib vector-export path should still work."""
    result = create_user_result()
    figure, _ = plot_source(
        result.reconstructed_source,
        grid=result.grid,
    )
    output_path = tmp_path / "reconstructed_source.pdf"

    try:
        figure.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(figure)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_reporting_rejects_invalid_result(
    tmp_path: Path,
) -> None:
    """Reporting functions should require a supported result."""
    with pytest.raises(ValidationError):
        export_results(
            "invalid result",  # type: ignore[arg-type]
            tmp_path,
        )


@pytest.mark.parametrize("invalid_dpi", [0, -10, 1.5, True])
def test_save_figures_rejects_invalid_dpi(
    tmp_path: Path,
    invalid_dpi: object,
) -> None:
    """Figure resolution must be a positive integer."""
    result = create_user_result()

    with pytest.raises(ValidationError):
        save_figures(
            result,
            tmp_path,
            dpi=invalid_dpi,  # type: ignore[arg-type]
        )


def test_scientific_exports_are_deterministic(tmp_path: Path) -> None:
    """Scientific CSV and JSON content should not depend on export time."""
    result = create_synthetic_result()
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"

    first_csv = pd.read_csv(save_metrics_csv(result, first_directory))
    second_csv = pd.read_csv(save_metrics_csv(result, second_directory))
    runtime_columns = [
        name for name in first_csv if "runtime" in name.lower()
    ]
    assert_frame_equal(
        first_csv.drop(columns=runtime_columns),
        second_csv.drop(columns=runtime_columns),
    )

    first_json = json.loads(
        save_config_json(result, first_directory).read_text(
            encoding="utf-8"
        )
    )
    second_json = json.loads(
        save_config_json(result, second_directory).read_text(
            encoding="utf-8"
        )
    )
    assert remove_runtime_values(first_json) == remove_runtime_values(
        second_json
    )
