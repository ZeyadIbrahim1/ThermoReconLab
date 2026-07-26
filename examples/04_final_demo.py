"""Run ThermoReconLab's presentation-ready end-to-end demonstration.

The three cases separate synthetic validation from reconstruction of
external measurements. Synthetic ground truth is used only for the two
benchmarks; the external workflow reports measurement-space diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from thermoreconlab.analysis import compute_error_field
from thermoreconlab.data import load_sensor_csv, save_array
from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
    reconstruct_from_measurements,
    run_regularization_study,
)
from thermoreconlab.reporting import export_results
from thermoreconlab.visualization import (
    plot_reconstruction_comparison,
    plot_regularization_study,
)


OUTPUT_DIRECTORY = Path("outputs") / "final_demo"
RECOVERABLE_DIRECTORY = OUTPUT_DIRECTORY / "recoverable_case"
SPARSE_DIRECTORY = OUTPUT_DIRECTORY / "sparse_stress_test"
USER_DIRECTORY = OUTPUT_DIRECTORY / "user_measurements"
USER_SENSOR_PATH = (
    Path("examples") / "data" / "demo_sensor_measurements.csv"
)

ALPHA_CANDIDATES = [
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
]
REGULARIZATION = "identity"
SEED = 42
FIGURE_DPI = 300

RECOVERABLE_GRID_SHAPE = (30, 30)
RECOVERABLE_SOURCE_TYPE = "gaussian"
RECOVERABLE_SENSOR_STRATEGY = "center_focused"
RECOVERABLE_NUM_SENSORS = 49
RECOVERABLE_NOISE_LEVEL = 0.02

SPARSE_GRID_SHAPE = (20, 20)
SPARSE_SOURCE_TYPE = "two_gaussians"
SPARSE_SENSOR_STRATEGY = "regular"
SPARSE_NUM_SENSORS = 16
SPARSE_NOISE_LEVEL = 0.02

USER_GRID_SHAPE = (20, 20)
USER_ALPHA = 1e-7
GROUND_TRUTH_UNAVAILABLE = (
    "Ground-truth source metrics are unavailable for this workflow."
)


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save one additional figure and close it even if saving fails."""
    try:
        figure.savefig(
            output_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    finally:
        plt.close(figure)


def interior_unknown_count(grid_shape: tuple[int, int]) -> int:
    """Return the number of source unknowns on interior grid nodes."""
    return (grid_shape[0] - 2) * (grid_shape[1] - 2)


def source_color_limits(result: ExperimentResult) -> tuple[float, float]:
    """Return common limits that preserve all true and recovered values."""
    lower = float(
        min(
            np.min(result.true_source),
            np.min(result.reconstructed_source),
        )
    )
    upper = float(
        max(
            np.max(result.true_source),
            np.max(result.reconstructed_source),
        )
    )

    if lower == upper:
        padding = 1.0 if lower == 0.0 else 0.05 * abs(lower)
        lower -= padding
        upper += padding

    return lower, upper


def select_tested_result(
    dataframe: pd.DataFrame,
    results: list[ExperimentResult],
) -> tuple[float, ExperimentResult]:
    """Select the tested synthetic result with the lowest source error."""
    error_values = dataframe["relative_l2_error"].to_numpy()
    selected_position = int(error_values.argmin())
    selected_alpha = float(dataframe.iloc[selected_position]["alpha"])
    return selected_alpha, results[selected_position]


def save_synthetic_arrays(
    result: ExperimentResult,
    output_directory: Path,
) -> None:
    """Save the four fields available in synthetic benchmark mode."""
    signed_error = compute_error_field(
        result.true_source,
        result.reconstructed_source,
    )
    save_array(output_directory / "true_source.npy", result.true_source)
    save_array(output_directory / "temperature.npy", result.temperature)
    save_array(
        output_directory / "reconstructed_source.npy",
        result.reconstructed_source,
    )
    save_array(output_directory / "signed_error.npy", signed_error)


def save_synthetic_summary(
    *,
    case_name: str,
    result: ExperimentResult,
    selected_alpha: float,
    output_directory: Path,
    interpretation_lines: list[str],
) -> None:
    """Save a concise presentation summary for one synthetic case."""
    unknowns = interior_unknown_count(result.grid.shape)
    measurements = result.reconstruction.n_sensors
    ratio = measurements / unknowns
    metrics = result.metrics
    lines = [
        case_name,
        "Mode: synthetic benchmark",
        f"Grid shape: {result.grid.shape}",
        f"Interior source unknowns: {unknowns}",
        f"Measurements: {measurements}",
        "Measurement-to-unknown ratio: "
        f"{measurements} / {unknowns} = {ratio:.6f}",
        f"Regularization: {REGULARIZATION}",
        f"Selected tested alpha: {selected_alpha:.3e}",
        "Alpha interpretation: Best tested alpha for this configuration.",
        f"RMSE: {metrics['rmse']:.6e}",
        f"MAE: {metrics['mae']:.6e}",
        f"Relative L2 source error: {metrics['relative_l2_error']:.6e}",
        "Maximum absolute source error: "
        f"{metrics['max_absolute_error']:.6e}",
        f"Residual norm: {metrics['residual_norm']:.6e}",
        f"Relative residual: {metrics['relative_residual']:.6e}",
        f"Residual RMS: {metrics['residual_rms']:.6e}",
        f"Solution norm: {metrics['solution_norm']:.6e}",
        "Source-error metrics are evaluated on interior source nodes.",
        *interpretation_lines,
        f"Output location: {output_directory}",
    ]
    (output_directory / "case_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def print_synthetic_result(
    case_name: str,
    result: ExperimentResult,
    selected_alpha: float,
) -> None:
    """Print the requested presentation metrics for a synthetic case."""
    unknowns = interior_unknown_count(result.grid.shape)
    measurements = result.reconstruction.n_sensors
    metrics = result.metrics
    print(f"\n{case_name}")
    print(f"  Grid shape: {result.grid.shape}")
    print(f"  Interior source unknowns: {unknowns}")
    print(f"  Measurements: {measurements}")
    print(
        "  Measurement-to-unknown ratio: "
        f"{measurements} / {unknowns} = {measurements / unknowns:.6f}"
    )
    print(f"  Regularization: {REGULARIZATION}")
    print(f"  Alpha: {selected_alpha:.3e}")
    print("  Best tested alpha for this configuration.")
    print(f"  RMSE: {metrics['rmse']:.6e}")
    print(f"  MAE: {metrics['mae']:.6e}")
    print(
        "  Relative L2 source error: "
        f"{metrics['relative_l2_error']:.6e}"
    )
    print(
        "  Maximum absolute source error: "
        f"{metrics['max_absolute_error']:.6e}"
    )
    print(f"  Residual norm: {metrics['residual_norm']:.6e}")
    print(f"  Relative residual: {metrics['relative_residual']:.6e}")
    print(f"  Residual RMS: {metrics['residual_rms']:.6e}")


def run_synthetic_case(
    *,
    case_name: str,
    output_directory: Path,
    grid_shape: tuple[int, int],
    source_type: str,
    sensor_strategy: str,
    num_sensors: int,
    noise_level: float,
    interpretation_lines: list[str],
) -> tuple[float, ExperimentResult]:
    """Run one compact identity-alpha study and export its selected result."""
    output_directory.mkdir(parents=True, exist_ok=True)
    dataframe, results = run_regularization_study(
        ALPHA_CANDIDATES,
        grid_shape=grid_shape,
        source_type=source_type,
        sensor_strategy=sensor_strategy,
        num_sensors=num_sensors,
        noise_level=noise_level,
        seed=SEED,
    )
    dataframe.to_csv(output_directory / "alpha_study.csv", index=False)

    alpha_figure, _ = plot_regularization_study(
        dataframe,
        metric="relative_l2_error",
        title=f"{case_name}: tested identity-alpha values",
    )
    save_figure(alpha_figure, output_directory / "alpha_study.png")

    selected_alpha, selected_result = select_tested_result(
        dataframe,
        results,
    )
    export_results(selected_result, output_directory, dpi=FIGURE_DPI)
    save_synthetic_arrays(selected_result, output_directory)

    source_vmin, source_vmax = source_color_limits(selected_result)
    comparison_figure, _ = plot_reconstruction_comparison(
        selected_result.grid,
        selected_result.true_source,
        selected_result.temperature,
        selected_result.sensor_data_noisy,
        selected_result.reconstructed_source,
        source_vmin=source_vmin,
        source_vmax=source_vmax,
        title=f"{case_name}: reconstruction comparison",
    )
    save_figure(
        comparison_figure,
        output_directory / "reconstruction_comparison.png",
    )
    save_synthetic_summary(
        case_name=case_name,
        result=selected_result,
        selected_alpha=selected_alpha,
        output_directory=output_directory,
        interpretation_lines=interpretation_lines,
    )
    print_synthetic_result(case_name, selected_result, selected_alpha)
    return selected_alpha, selected_result


def run_recoverable_case() -> None:
    """Run Case A, a reasonably informative synthetic benchmark."""
    run_synthetic_case(
        case_name="Case A — Recoverable synthetic benchmark",
        output_directory=RECOVERABLE_DIRECTORY,
        grid_shape=RECOVERABLE_GRID_SHAPE,
        source_type=RECOVERABLE_SOURCE_TYPE,
        sensor_strategy=RECOVERABLE_SENSOR_STRATEGY,
        num_sensors=RECOVERABLE_NUM_SENSORS,
        noise_level=RECOVERABLE_NOISE_LEVEL,
        interpretation_lines=[
            "Synthetic truth permits benchmark-specific validation.",
            "The sensing configuration is designed to be reasonably "
            "informative, not perfect.",
            "The tested alpha values are specific to this configuration.",
            "No universal reconstruction accuracy or real experimental "
            "validation is claimed.",
        ],
    )


def run_sparse_stress_test() -> None:
    """Run Case B, a deliberately severe undersampling benchmark."""
    run_synthetic_case(
        case_name="Case B — Sparse stress test",
        output_directory=SPARSE_DIRECTORY,
        grid_shape=SPARSE_GRID_SHAPE,
        source_type=SPARSE_SOURCE_TYPE,
        sensor_strategy=SPARSE_SENSOR_STRATEGY,
        num_sensors=SPARSE_NUM_SENSORS,
        noise_level=SPARSE_NOISE_LEVEL,
        interpretation_lines=[
            "This is a deliberate sparse stress test.",
            "Broad or incomplete recovery is expected.",
            "A low sensor-space residual does not prove accurate source "
            "recovery.",
            "This is not a failure hidden by visual scaling; common source "
            "color limits are used.",
        ],
    )
    print("  This is a deliberate sparse stress test.")
    print(
        "  A low sensor-space residual does not prove accurate source "
        "recovery."
    )
    print("  Broad or incomplete recovery is expected.")
    print(
        "  This is not a failure hidden by visual scaling; common source "
        "color limits are used."
    )


def save_user_summary(
    result: MeasurementReconstructionResult,
    output_directory: Path,
) -> None:
    """Save the presentation summary for external measurements."""
    unknowns = interior_unknown_count(result.grid.shape)
    measurements = result.reconstruction.n_sensors
    metrics = result.metrics
    lines = [
        "Case C — External user-measurement workflow",
        "Mode: user measurements",
        f"Grid shape: {result.grid.shape}",
        f"Interior source unknowns: {unknowns}",
        f"Measurements: {measurements}",
        "Measurement-to-unknown ratio: "
        f"{measurements} / {unknowns} = {measurements / unknowns:.6f}",
        f"Regularization: {REGULARIZATION}",
        f"Alpha: {USER_ALPHA:.3e}",
        "Alpha interpretation: Fixed demonstration setting for this dataset; "
        "it is not automatically selected or universally optimal.",
        f"Residual norm: {metrics['residual_norm']:.6e}",
        f"Relative residual: {metrics['relative_residual']:.6e}",
        f"Residual RMS: {metrics['residual_rms']:.6e}",
        f"Solution norm: {metrics['solution_norm']:.6e}",
        GROUND_TRUTH_UNAVAILABLE,
        "No real experimental validation is claimed.",
        f"Output location: {output_directory}",
    ]
    (output_directory / "case_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_user_measurement_case() -> None:
    """Run Case C without generating or accessing synthetic truth."""
    USER_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sensor_data = load_sensor_csv(USER_SENSOR_PATH)
    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=USER_GRID_SHAPE,
        alpha=USER_ALPHA,
        regularization=REGULARIZATION,
    )
    export_results(result, USER_DIRECTORY, dpi=FIGURE_DPI)
    save_array(
        USER_DIRECTORY / "reconstructed_source.npy",
        result.reconstructed_source,
    )
    save_user_summary(result, USER_DIRECTORY)

    unknowns = interior_unknown_count(result.grid.shape)
    measurements = result.reconstruction.n_sensors
    metrics = result.metrics
    print("\nCase C — External user-measurement workflow")
    print(f"  Grid shape: {result.grid.shape}")
    print(f"  Interior source unknowns: {unknowns}")
    print(f"  Measurements: {measurements}")
    print(
        "  Measurement-to-unknown ratio: "
        f"{measurements} / {unknowns} = {measurements / unknowns:.6f}"
    )
    print(f"  Regularization: {REGULARIZATION}")
    print(f"  Alpha: {USER_ALPHA:.3e} (fixed demonstration setting)")
    print(f"  Residual norm: {metrics['residual_norm']:.6e}")
    print(f"  Relative residual: {metrics['relative_residual']:.6e}")
    print(f"  Residual RMS: {metrics['residual_rms']:.6e}")
    print(f"  Solution norm: {metrics['solution_norm']:.6e}")
    print(GROUND_TRUTH_UNAVAILABLE)


def print_scientific_context() -> None:
    """Print the interpretation boundaries shared by all three cases."""
    print("\nScientific context")
    print(
        "  Synthetic data supports validation; source-error metrics use "
        "interior source nodes."
    )
    print(
        "  Residuals are measured in sensor space, so a low residual does "
        "not guarantee source accuracy."
    )
    print(
        "  Sparse inverse reconstruction is underdetermined; sensor geometry "
        "and regularization both matter."
    )
    print(
        "  Tested alpha values are configuration-specific benchmark choices, "
        "not automatic package selections."
    )
    print(
        "  External user data has no known ground truth, and no real "
        "experimental validation is claimed."
    )


def main() -> None:
    """Run the three final demonstration cases."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print("ThermoReconLab final demonstration")
    run_recoverable_case()
    run_sparse_stress_test()
    run_user_measurement_case()
    print_scientific_context()
    print(f"\nFinal output directory: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
