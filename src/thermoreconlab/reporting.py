"""Result export and reporting tools for ThermoReconLab.

This module keeps numerical summaries, configuration export, figures,
and Markdown reporting in one cohesive architectural component.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd

from thermoreconlab.analysis import compute_error_field
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
)
from thermoreconlab.visualization import (
    plot_error_map,
    plot_sensor_measurements,
    plot_source,
    plot_temperature,
)

SupportedResult = ExperimentResult | MeasurementReconstructionResult


def _validate_result(result: SupportedResult) -> None:
    """Validate a supported experiment result object."""
    if not isinstance(
        result,
        (ExperimentResult, MeasurementReconstructionResult),
    ):
        raise ValidationError(
            "result must be an ExperimentResult or "
            "MeasurementReconstructionResult object."
        )


def _prepare_output_directory(output_dir: str | Path) -> Path:
    """Create and return an output directory."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_config_json(
    result: SupportedResult,
    output_dir: str | Path,
) -> Path:
    """Save experiment settings and solver diagnostics as JSON."""
    _validate_result(result)
    directory = _prepare_output_directory(output_dir)
    output_path = directory / "summary.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result.to_dict(),
            file,
            indent=2,
            allow_nan=False,
        )

    return output_path


def save_metrics_csv(
    result: SupportedResult,
    output_dir: str | Path,
) -> Path:
    """Save metrics and reconstruction diagnostics to one CSV row."""
    _validate_result(result)
    directory = _prepare_output_directory(output_dir)
    output_path = directory / "metrics.csv"

    row: dict[str, Any] = {
        "mode": result.config["mode"],
        "grid_nx": result.grid.nx,
        "grid_ny": result.grid.ny,
        "num_sensors": result.reconstruction.n_sensors,
        "alpha": result.reconstruction.alpha,
        "residual_norm": result.reconstruction.residual_norm,
        "solution_norm": result.reconstruction.solution_norm,
        "reconstruction_runtime": result.reconstruction.runtime,
        "total_runtime": result.runtime,
    }

    if isinstance(result, ExperimentResult):
        row.update(result.metrics)

    pd.DataFrame([row]).to_csv(output_path, index=False)

    return output_path


def save_figures(
    result: SupportedResult,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Path]:
    """Save the standard scientific figures for a result."""
    _validate_result(result)

    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
        raise ValidationError("dpi must be a positive integer.")

    directory = _prepare_output_directory(output_dir)
    figure_directory = directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}

    def save_figure(name: str, figure: Any) -> None:
        path = figure_directory / f"{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved_paths[name] = path

    figure, _ = plot_source(
        result.reconstructed_source,
        grid=result.grid,
        title="Reconstructed source",
    )
    save_figure("reconstructed_source", figure)

    measurement_data = (
        result.sensor_data_noisy
        if isinstance(result, ExperimentResult)
        else result.sensor_data
    )

    figure, _ = plot_sensor_measurements(
        measurement_data,
        result.grid,
        title="Sensor measurements",
    )
    save_figure("sensor_measurements", figure)

    if isinstance(result, ExperimentResult):
        figure, _ = plot_source(
            result.true_source,
            grid=result.grid,
            title="True source",
        )
        save_figure("true_source", figure)

        figure, _ = plot_temperature(
            result.temperature,
            grid=result.grid,
            title="Temperature field",
        )
        save_figure("temperature", figure)

        error_field = compute_error_field(
            result.true_source,
            result.reconstructed_source,
        )
        figure, _ = plot_error_map(
            error_field,
            grid=result.grid,
            title="Reconstruction error",
        )
        save_figure("error_map", figure)

    return saved_paths


def write_markdown_report(
    result: SupportedResult,
    output_dir: str | Path,
    *,
    figure_paths: dict[str, Path] | None = None,
) -> Path:
    """Write a concise Markdown report for one reconstruction."""
    _validate_result(result)
    directory = _prepare_output_directory(output_dir)
    output_path = directory / "report.md"

    lines = [
        "# ThermoReconLab Result Report",
        "",
        f"- **Mode:** {result.config['mode']}",
        f"- **Grid:** {result.grid.nx} × {result.grid.ny}",
        f"- **Sensors:** {result.reconstruction.n_sensors}",
        f"- **Regularization parameter:** "
        f"{result.reconstruction.alpha:.3e}",
        f"- **Residual norm:** "
        f"{result.reconstruction.residual_norm:.6e}",
        f"- **Solution norm:** "
        f"{result.reconstruction.solution_norm:.6e}",
        f"- **Total runtime:** {result.runtime:.6f} s",
        "",
    ]

    if isinstance(result, ExperimentResult):
        lines.extend(
            [
                "## Ground-truth validation",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )

        for name, value in result.metrics.items():
            readable_name = name.replace("_", " ").title()
            lines.append(f"| {readable_name} | {value:.6e} |")

        lines.extend(
            [
                "",
                "These metrics are available because synthetic "
                "benchmark mode provides a known true source.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Validation note",
                "",
                "Ground-truth source metrics are not reported because "
                "the true source is unknown for user-provided "
                "measurements.",
                "",
            ]
        )

    if figure_paths:
        lines.extend(["## Figures", ""])

        for name, path in figure_paths.items():
            relative_path = path.relative_to(directory)
            readable_name = name.replace("_", " ").title()
            lines.append(f"### {readable_name}")
            lines.append("")
            lines.append(
                f"![{readable_name}]({relative_path.as_posix()})"
            )
            lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            "- Two-dimensional steady-state model.",
            "- Homogeneous Dirichlet boundary conditions.",
            "- Structured finite-difference grid.",
            "- Identity Tikhonov regularization.",
            "- User-data accuracy cannot be quantified without ground truth.",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_path


def export_results(
    result: SupportedResult,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Any]:
    """Export the complete standard result package."""
    _validate_result(result)
    directory = _prepare_output_directory(output_dir)

    figures = save_figures(result, directory, dpi=dpi)
    metrics_path = save_metrics_csv(result, directory)
    summary_path = save_config_json(result, directory)
    report_path = write_markdown_report(
        result,
        directory,
        figure_paths=figures,
    )

    return {
        "output_dir": directory,
        "metrics": metrics_path,
        "summary": summary_path,
        "report": report_path,
        "figures": figures,
    }
