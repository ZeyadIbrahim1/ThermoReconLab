"""Result export and reporting tools for ThermoReconLab.

This module keeps numerical summaries, configuration export, figures,
and Markdown reporting in one cohesive architectural component.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from thermoreconlab.analysis import (
    analyze_observation_matrix,
    compute_error_field,
)
from thermoreconlab.exceptions import ValidationError
from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
)
from thermoreconlab.reconstruction import build_observation_matrix
from thermoreconlab.sensors import SensorData
from thermoreconlab.visualization import (
    plot_error_map,
    plot_reconstruction_comparison,
    plot_sensor_measurements,
    plot_source,
    plot_temperature,
)

SupportedResult = ExperimentResult | MeasurementReconstructionResult

_COMMON_CSV_FIELDS = (
    "mode",
    "grid_nx",
    "grid_ny",
    "num_sensors",
    "num_measurements",
    "num_unknowns",
    "measurement_to_unknown_ratio",
    "regularization",
    "alpha",
    "residual_norm",
    "relative_residual",
    "residual_rms",
    "solution_norm",
    "reconstruction_runtime",
    "total_runtime",
)
_SYNTHETIC_CSV_FIELDS = (
    "source_type",
    "sensor_strategy",
    "noise_level",
    "seed",
    "source_metric_domain",
    "rmse",
    "mae",
    "relative_l2_error",
    "max_absolute_error",
)
_OBSERVATION_CSV_FIELDS = (
    "numerical_rank",
    "nullity",
    "is_underdetermined",
    "largest_singular_value",
    "smallest_resolved_singular_value",
    "effective_condition_number",
    "rank_tolerance",
)
_FIGURE_ORDER = (
    "true_source",
    "temperature",
    "sensor_measurements",
    "reconstructed_source",
    "error_map",
    "reconstruction_comparison",
)


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


def _validate_dpi(dpi: int) -> None:
    """Validate a raster export resolution."""
    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
        raise ValidationError("dpi must be a positive integer.")


def _measurement_data(result: SupportedResult) -> SensorData:
    """Return the measurements actually used by a reconstruction."""
    if isinstance(result, ExperimentResult):
        return result.sensor_data_noisy

    return result.sensor_data


def _collect_reporting_summary(
    result: SupportedResult,
) -> dict[str, Any]:
    """Collect shared, mode-aware values for standard reports."""
    metrics = result.metrics
    num_measurements = int(result.reconstruction.n_sensors)
    num_unknowns = (result.grid.nx - 2) * (result.grid.ny - 2)
    measurement_data = _measurement_data(result)
    observation_matrix = build_observation_matrix(
        measurement_data.indices,
        result.grid,
    )
    diagnostics = analyze_observation_matrix(observation_matrix)

    summary: dict[str, Any] = {
        "mode": result.config["mode"],
        "grid_nx": int(result.grid.nx),
        "grid_ny": int(result.grid.ny),
        "grid_shape": [int(result.grid.nx), int(result.grid.ny)],
        "num_sensors": num_measurements,
        "num_measurements": num_measurements,
        "num_unknowns": num_unknowns,
        "measurement_to_unknown_ratio": (
            num_measurements / num_unknowns
        ),
        "regularization": result.config["regularization"],
        "alpha": float(result.reconstruction.alpha),
        "residual_norm": float(metrics["residual_norm"]),
        "relative_residual": float(metrics["relative_residual"]),
        "residual_rms": float(metrics["residual_rms"]),
        "solution_norm": float(metrics["solution_norm"]),
        "reconstruction_runtime": float(result.reconstruction.runtime),
        "total_runtime": float(result.runtime),
        "observation_matrix_shape": [
            int(diagnostics["shape"][0]),
            int(diagnostics["shape"][1]),
        ],
        "numerical_rank": int(diagnostics["numerical_rank"]),
        "nullity": int(diagnostics["nullity"]),
        "is_underdetermined": bool(diagnostics["underdetermined"]),
        "largest_singular_value": float(
            diagnostics["largest_singular_value"]
        ),
        "smallest_resolved_singular_value": float(
            diagnostics["smallest_resolved_singular_value"]
        ),
        "rank_tolerance": float(diagnostics["rank_tolerance"]),
    }

    condition_number = float(
        diagnostics["effective_condition_number"]
    )
    if math.isfinite(condition_number):
        summary["effective_condition_number"] = condition_number

    for name in (
        "source_type",
        "sensor_strategy",
        "noise_level",
        "seed",
        "source_metric_domain",
        "measurement_selection",
    ):
        if name in result.config and result.config[name] is not None:
            summary[name] = result.config[name]

    if isinstance(result, ExperimentResult):
        for name in (
            "rmse",
            "mae",
            "relative_l2_error",
            "max_absolute_error",
        ):
            summary[name] = float(metrics[name])

    return summary


def _json_safe(value: Any) -> Any:
    """Represent non-finite floats explicitly for strict JSON output."""
    if isinstance(value, dict):
        return {
            str(name): _json_safe(item)
            for name, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def _write_config_json(
    result: SupportedResult,
    directory: Path,
    reporting_summary: dict[str, Any],
) -> Path:
    """Write an already validated result summary."""
    output_path = directory / "summary.json"
    content = result.to_dict()
    content["metrics"] = dict(result.metrics)
    content["reporting_summary"] = dict(reporting_summary)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(content),
            file,
            indent=2,
            allow_nan=False,
        )

    return output_path


def save_config_json(
    result: SupportedResult,
    output_dir: str | Path,
) -> Path:
    """Save experiment settings and solver diagnostics as JSON."""
    _validate_result(result)
    directory = _prepare_output_directory(output_dir)
    reporting_summary = _collect_reporting_summary(result)
    return _write_config_json(result, directory, reporting_summary)


def _write_metrics_csv(
    directory: Path,
    reporting_summary: dict[str, Any],
) -> Path:
    """Write one deterministic metrics row from collected values."""
    output_path = directory / "metrics.csv"
    field_order = (
        *_COMMON_CSV_FIELDS,
        *_SYNTHETIC_CSV_FIELDS,
        *_OBSERVATION_CSV_FIELDS,
        "measurement_selection",
    )
    row = {
        name: reporting_summary[name]
        for name in field_order
        if name in reporting_summary
    }
    pd.DataFrame([row], columns=list(row)).to_csv(
        output_path,
        index=False,
    )
    return output_path


def save_metrics_csv(
    result: SupportedResult,
    output_dir: str | Path,
) -> Path:
    """Save metrics and reconstruction diagnostics to one CSV row."""
    _validate_result(result)
    directory = _prepare_output_directory(output_dir)
    reporting_summary = _collect_reporting_summary(result)
    return _write_metrics_csv(directory, reporting_summary)


def save_figures(
    result: SupportedResult,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Path]:
    """Save the standard scientific figures for a result."""
    _validate_result(result)
    _validate_dpi(dpi)

    directory = _prepare_output_directory(output_dir)
    figure_directory = directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}

    def save_figure(name: str, figure: Figure) -> None:
        path = figure_directory / f"{name}.png"
        try:
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
        finally:
            plt.close(figure)
        saved_paths[name] = path

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

    measurement_data = _measurement_data(result)
    figure, _ = plot_sensor_measurements(
        measurement_data,
        result.grid,
        title="Sensor measurements",
    )
    save_figure("sensor_measurements", figure)

    figure, _ = plot_source(
        result.reconstructed_source,
        grid=result.grid,
        title="Reconstructed source",
    )
    save_figure("reconstructed_source", figure)

    if isinstance(result, ExperimentResult):
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

        figure, _ = plot_reconstruction_comparison(
            result.grid,
            result.true_source,
            result.temperature,
            result.sensor_data_noisy,
            result.reconstructed_source,
        )
        save_figure("reconstruction_comparison", figure)

    return saved_paths


def _observation_report_lines(
    reporting_summary: dict[str, Any],
) -> list[str]:
    """Return concise observation-matrix diagnostics for Markdown."""
    shape = reporting_summary["observation_matrix_shape"]
    lines = [
        "## Observation-matrix diagnostics",
        "",
        f"- **Shape:** {shape[0]} × {shape[1]}",
        f"- **Numerical rank:** {reporting_summary['numerical_rank']}",
        f"- **Nullity:** {reporting_summary['nullity']}",
        "- **Underdetermined:** "
        f"{reporting_summary['is_underdetermined']}",
    ]
    if "effective_condition_number" in reporting_summary:
        lines.append(
            "- **Effective condition number:** "
            f"{reporting_summary['effective_condition_number']:.6e}"
        )
    lines.extend(
        [
            "",
            "Numerical rank is limited by the number of measurements; "
            "cases with fewer measurements than unknowns remain "
            "underdetermined.",
            "",
        ]
    )
    return lines


def _synthetic_report_lines(
    reporting_summary: dict[str, Any],
) -> list[str]:
    """Return synthetic-only configuration and validation sections."""
    lines = ["## Synthetic benchmark configuration", ""]
    labels = {
        "source_type": "Source type",
        "sensor_strategy": "Sensor strategy",
        "noise_level": "Noise level",
        "seed": "Seed",
    }
    for name, label in labels.items():
        if name in reporting_summary:
            lines.append(f"- **{label}:** {reporting_summary[name]}")

    lines.extend(
        [
            "",
            "## Ground-truth validation",
            "",
            "| Source metric | Value |",
            "|---|---:|",
            f"| RMSE | {reporting_summary['rmse']:.6e} |",
            f"| MAE | {reporting_summary['mae']:.6e} |",
            "| Relative L2 Error | "
            f"{reporting_summary['relative_l2_error']:.6e} |",
            "| Maximum Absolute Error | "
            f"{reporting_summary['max_absolute_error']:.6e} |",
            "",
            "Source-error metrics are evaluated on interior source nodes.",
            "Synthetic truth supports validation and benchmarking; this is "
            "not real experimental validation.",
            "",
        ]
    )
    return lines


def _measurement_report_lines() -> list[str]:
    """Return validation context for workflows without source truth."""
    return [
        "## Measurement-space diagnostics",
        "",
        "Ground-truth source metrics are unavailable for this workflow.",
        "Residual agreement is evaluated in measurement space and does not "
        "by itself establish source accuracy.",
        "",
    ]


def _figure_report_lines(
    directory: Path,
    figure_paths: dict[str, Path],
) -> list[str]:
    """Return deterministic relative links for available figures."""
    available_names = set(figure_paths)
    ordered_names = [
        name for name in _FIGURE_ORDER if name in available_names
    ]
    ordered_names.extend(sorted(available_names.difference(ordered_names)))
    lines = ["## Figures", ""]

    for name in ordered_names:
        path = figure_paths[name]
        relative_path = path.relative_to(directory)
        readable_name = name.replace("_", " ").title()
        lines.extend(
            [
                f"### {readable_name}",
                "",
                f"![{readable_name}]({relative_path.as_posix()})",
                "",
            ]
        )

    return lines


def _limitation_report_lines(is_synthetic: bool) -> list[str]:
    """Return concise limitations appropriate to the result mode."""
    lines = [
        "## Limitations",
        "",
        "- Two-dimensional steady-state model.",
        "- Homogeneous Dirichlet boundary assumption.",
    ]
    if is_synthetic:
        lines.extend(
            [
                "- With sparse data, reconstruction remains underdetermined.",
                "- Results depend on sensor geometry and the selected alpha.",
                "- Amplitudes may be reduced by regularization.",
                "- Negative reconstruction artifacts may occur.",
                "- Residuals are evaluated in sensor space; a low residual "
                "does not guarantee a low source error.",
            ]
        )
    else:
        lines.extend(
            [
                "- Ground-truth source values are not known.",
                "- Residual agreement does not prove source accuracy.",
                "- Results depend on the forward model.",
                "- Results depend on sensor or measurement geometry.",
                "- Results depend on the selected regularization and alpha.",
            ]
        )
    lines.append("")
    return lines


def _write_markdown_report(
    result: SupportedResult,
    directory: Path,
    reporting_summary: dict[str, Any],
    figure_paths: dict[str, Path] | None,
) -> Path:
    """Write a report from already collected summary values."""
    output_path = directory / "report.md"
    lines = [
        "# ThermoReconLab Result Report",
        "",
        f"- **Mode:** {reporting_summary['mode']}",
        "- **Grid:** "
        f"{reporting_summary['grid_nx']} × "
        f"{reporting_summary['grid_ny']}",
        f"- **Measurements:** {reporting_summary['num_measurements']}",
        "- **Interior source unknowns:** "
        f"{reporting_summary['num_unknowns']}",
        "- **Measurement-to-unknown ratio:** "
        f"{reporting_summary['measurement_to_unknown_ratio']:.6f}",
        f"- **Regularization:** {reporting_summary['regularization']}",
        f"- **Alpha:** {reporting_summary['alpha']:.3e}",
        f"- **Residual norm:** {reporting_summary['residual_norm']:.6e}",
        "- **Relative residual:** "
        f"{reporting_summary['relative_residual']:.6e}",
        f"- **Residual RMS:** {reporting_summary['residual_rms']:.6e}",
        f"- **Solution norm:** {reporting_summary['solution_norm']:.6e}",
        "- **Reconstruction runtime:** "
        f"{reporting_summary['reconstruction_runtime']:.6f} s",
        f"- **Total runtime:** {reporting_summary['total_runtime']:.6f} s",
        "",
    ]
    lines.extend(_observation_report_lines(reporting_summary))

    is_synthetic = isinstance(result, ExperimentResult)
    if is_synthetic:
        lines.extend(_synthetic_report_lines(reporting_summary))
    else:
        lines.extend(_measurement_report_lines())

    if figure_paths:
        lines.extend(_figure_report_lines(directory, figure_paths))

    lines.extend(_limitation_report_lines(is_synthetic))
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_markdown_report(
    result: SupportedResult,
    output_dir: str | Path,
    *,
    figure_paths: dict[str, Path] | None = None,
) -> Path:
    """Write a concise Markdown report for one reconstruction."""
    _validate_result(result)
    directory = _prepare_output_directory(output_dir)
    reporting_summary = _collect_reporting_summary(result)
    return _write_markdown_report(
        result,
        directory,
        reporting_summary,
        figure_paths,
    )


def export_results(
    result: SupportedResult,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Any]:
    """Export the complete standard result package."""
    _validate_result(result)
    _validate_dpi(dpi)
    directory = _prepare_output_directory(output_dir)
    reporting_summary = _collect_reporting_summary(result)
    metrics_path = _write_metrics_csv(directory, reporting_summary)
    summary_path = _write_config_json(
        result,
        directory,
        reporting_summary,
    )
    figures = save_figures(result, directory, dpi=dpi)
    report_path = _write_markdown_report(
        result,
        directory,
        reporting_summary,
        figures,
    )

    return {
        "output_dir": directory,
        "metrics": metrics_path,
        "summary": summary_path,
        "report": report_path,
        "figures": figures,
    }
