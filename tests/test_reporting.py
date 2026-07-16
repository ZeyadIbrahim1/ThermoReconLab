"""Tests for result reporting and export."""

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from thermoreconlab import (
    reconstruct_from_measurements,
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
from thermoreconlab.sensors import (
    create_sensor_data,
    regular_grid_sensors,
)
from thermoreconlab.visualization import plot_source


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


def test_save_config_json_creates_file(tmp_path: Path) -> None:
    """JSON export should create a readable summary file."""
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        seed=42,
    )

    path = save_config_json(result, tmp_path)

    assert path.exists()
    assert path.name == "summary.json"
    assert '"mode": "synthetic_benchmark"' in path.read_text(
        encoding="utf-8"
    )


def test_save_metrics_csv_creates_expected_columns(
    tmp_path: Path,
) -> None:
    """Synthetic CSV export should contain validation metrics."""
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        seed=42,
    )

    path = save_metrics_csv(result, tmp_path)
    dataframe = pd.read_csv(path)

    assert path.exists()
    assert len(dataframe) == 1
    assert "relative_l2_error" in dataframe.columns
    assert "residual_norm" in dataframe.columns
    assert dataframe.loc[0, "mode"] == "synthetic_benchmark"


def test_user_metrics_csv_omits_ground_truth_metrics(
    tmp_path: Path,
) -> None:
    """User results should not claim unavailable source errors."""
    result = create_user_result()

    path = save_metrics_csv(result, tmp_path)
    dataframe = pd.read_csv(path)

    assert "residual_norm" in dataframe.columns
    assert "relative_l2_error" not in dataframe.columns
    assert dataframe.loc[0, "mode"] == "user_measurements"


def test_save_figures_creates_synthetic_figure_set(
    tmp_path: Path,
) -> None:
    """Synthetic export should create five standard figures."""
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        seed=42,
    )

    paths = save_figures(result, tmp_path, dpi=80)

    assert set(paths) == {
        "reconstructed_source",
        "sensor_measurements",
        "true_source",
        "temperature",
        "error_map",
    }
    assert all(path.exists() for path in paths.values())


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


def test_markdown_report_creates_file(
    tmp_path: Path,
) -> None:
    """Markdown export should contain the report title."""
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        seed=42,
    )

    path = write_markdown_report(result, tmp_path)

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "# ThermoReconLab Result Report" in content
    assert "Ground-truth validation" in content


def test_export_results_creates_complete_package(
    tmp_path: Path,
) -> None:
    """Full export should create tables, report, and figures."""
    result = run_synthetic_benchmark(
        grid_shape=(8, 8),
        num_sensors=9,
        seed=42,
    )

    exported = export_results(result, tmp_path, dpi=80)

    assert exported["metrics"].exists()
    assert exported["summary"].exists()
    assert exported["report"].exists()
    assert exported["output_dir"] == tmp_path
    assert all(
        path.exists()
        for path in exported["figures"].values()
    )


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

    figure.savefig(output_path, bbox_inches="tight")

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
