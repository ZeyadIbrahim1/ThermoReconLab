"""Run reproducible parameter studies and inverse-problem diagnostics.

This official example uses synthetic data to compare controlled study
settings. It saves compact CSV, JSON, and PNG outputs under
``outputs/parameter_studies`` without changing package functionality.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from thermoreconlab.analysis import analyze_observation_matrix
from thermoreconlab.experiments import (
    ExperimentResult,
    run_regularization_comparison,
    run_regularization_study,
    run_repeated_noise_study,
    run_sensor_count_study,
    run_sensor_layout_study,
)
from thermoreconlab.reconstruction import build_observation_matrix
from thermoreconlab.visualization import (
    plot_regularization_study,
    plot_repeated_noise_study,
    plot_sensor_count_study,
    plot_sensor_layout_study,
    plot_singular_values,
)


OUTPUT_DIRECTORY = Path("outputs/parameter_studies")

GRID_SHAPE = (20, 20)
SOURCE_TYPE = "two_gaussians"
SENSOR_STRATEGY = "regular"
NUM_SENSORS = 25
NOISE_LEVEL = 0.02
MASTER_SEED = 42

IDENTITY_ALPHA_VALUES = [
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
]
SENSOR_COUNTS = [4, 9, 16, 25, 36]
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10]
REPEATED_SEEDS = [10, 20, 30, 40, 50]
SENSOR_LAYOUTS = ["regular", "random", "center_focused"]
SMOOTHNESS_ALPHA = 1e-9


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save one report-quality figure and immediately close it."""
    try:
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
    finally:
        plt.close(figure)


def _is_nonfinite_number(value: Any) -> bool:
    """Return whether a scalar numeric diagnostic is NaN or infinite."""
    if isinstance(value, (bool, np.bool_)):
        return False

    if isinstance(value, (float, np.floating)):
        return not math.isfinite(float(value))

    return False


def _nonfinite_note(name: str, value: Any) -> str:
    """Explain why one non-finite diagnostic is represented by null."""
    if name == "effective_condition_number" and math.isinf(float(value)):
        return (
            "Infinite because no resolved nonzero singular value was "
            "available."
        )

    return (
        "The diagnostic was non-finite and is stored as null because "
        "JSON does not support NaN or Infinity."
    )


def to_json_safe(value: Any) -> Any:
    """Recursively convert NumPy and non-finite diagnostics for JSON."""
    if isinstance(value, dict):
        converted: dict[str, Any] = {}

        for raw_name, item in value.items():
            name = str(raw_name)
            converted[name] = to_json_safe(item)

            if _is_nonfinite_number(item):
                converted[f"{name}_note"] = _nonfinite_note(name, item)

        return converted

    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())

    if isinstance(value, tuple):
        return [to_json_safe(item) for item in value]

    if isinstance(value, list):
        return [to_json_safe(item) for item in value]

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def print_reference_configuration() -> None:
    """Print the shared configuration used by the example."""
    print("ThermoReconLab parameter-study example")
    print("Reference configuration:")
    print(f"  Grid shape: {GRID_SHAPE}")
    print(f"  Source type: {SOURCE_TYPE}")
    print(f"  Sensor strategy: {SENSOR_STRATEGY}")
    print(f"  Number of sensors: {NUM_SENSORS}")
    print(f"  Noise level: {NOISE_LEVEL:.2f}")
    print(f"  Master seed: {MASTER_SEED}")


def run_identity_alpha_study() -> tuple[float, ExperimentResult]:
    """Run the identity-Tikhonov alpha study and save its outputs."""
    dataframe, results = run_regularization_study(
        IDENTITY_ALPHA_VALUES,
        grid_shape=GRID_SHAPE,
        source_type=SOURCE_TYPE,
        sensor_strategy=SENSOR_STRATEGY,
        num_sensors=NUM_SENSORS,
        noise_level=NOISE_LEVEL,
        seed=MASTER_SEED,
    )
    dataframe.to_csv(
        OUTPUT_DIRECTORY / "regularization_study.csv",
        index=False,
    )

    figure, _ = plot_regularization_study(
        dataframe,
        metric="relative_l2_error",
    )
    save_figure(
        figure,
        OUTPUT_DIRECTORY / "regularization_study.png",
    )

    selected_position = int(
        dataframe["relative_l2_error"].to_numpy().argmin()
    )
    selected_alpha = float(
        dataframe.iloc[selected_position]["alpha"]
    )
    selected_result = results[selected_position]

    print(
        "\nBest tested value for this configuration: "
        f"{selected_alpha:.3e}"
    )
    return selected_alpha, selected_result


def run_sensor_count_example(identity_alpha: float) -> None:
    """Study sensor count using one fixed, tested identity alpha."""
    dataframe, _ = run_sensor_count_study(
        SENSOR_COUNTS,
        grid_shape=GRID_SHAPE,
        source_type=SOURCE_TYPE,
        sensor_strategy=SENSOR_STRATEGY,
        noise_level=NOISE_LEVEL,
        alpha=identity_alpha,
        seed=MASTER_SEED,
    )
    dataframe.to_csv(
        OUTPUT_DIRECTORY / "sensor_count_study.csv",
        index=False,
    )

    figure, _ = plot_sensor_count_study(
        dataframe,
        metric="relative_l2_error",
    )
    save_figure(
        figure,
        OUTPUT_DIRECTORY / "sensor_count_study.png",
    )

    print("\nSensor-count study:")
    for row in dataframe.itertuples(index=False):
        print(
            f"  sensors={row.sensor_count:2d}, "
            f"relative L2 source error={row.relative_l2_error:.6f}"
        )
    print("  Sensor-count effects are configuration-dependent.")


def run_repeated_noise_example(identity_alpha: float) -> None:
    """Study repeated noise realizations at controlled noise levels."""
    detailed, summary, _ = run_repeated_noise_study(
        NOISE_LEVELS,
        REPEATED_SEEDS,
        grid_shape=GRID_SHAPE,
        source_type=SOURCE_TYPE,
        sensor_strategy=SENSOR_STRATEGY,
        num_sensors=NUM_SENSORS,
        alpha=identity_alpha,
        seed=MASTER_SEED,
    )
    detailed.to_csv(
        OUTPUT_DIRECTORY / "repeated_noise_detailed.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIRECTORY / "repeated_noise_summary.csv",
        index=False,
    )

    figure, _ = plot_repeated_noise_study(
        summary,
    )
    save_figure(
        figure,
        OUTPUT_DIRECTORY / "repeated_noise_study.png",
    )

    print("\nRepeated-noise study:")
    for row in summary.itertuples(index=False):
        print(
            f"  noise={row.noise_level:.2f}, "
            f"mean relative L2 source error="
            f"{row.mean_relative_l2_error:.6f}, "
            f"population std={row.std_relative_l2_error:.6f}"
        )


def run_sensor_layout_example(identity_alpha: float) -> None:
    """Compare sensor layouts over repeated deterministic run seeds."""
    detailed, summary, _ = run_sensor_layout_study(
        SENSOR_LAYOUTS,
        REPEATED_SEEDS,
        grid_shape=GRID_SHAPE,
        source_type=SOURCE_TYPE,
        num_sensors=NUM_SENSORS,
        noise_level=NOISE_LEVEL,
        alpha=identity_alpha,
        seed=MASTER_SEED,
    )
    detailed.to_csv(
        OUTPUT_DIRECTORY / "sensor_layout_detailed.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIRECTORY / "sensor_layout_summary.csv",
        index=False,
    )

    figure, _ = plot_sensor_layout_study(
        summary,
    )
    save_figure(
        figure,
        OUTPUT_DIRECTORY / "sensor_layout_study.png",
    )

    print("\nSensor-layout study:")
    for row in summary.itertuples(index=False):
        print(
            f"  strategy={row.strategy}, "
            f"mean relative L2 source error="
            f"{row.mean_relative_l2_error:.6f}, "
            f"population std={row.std_relative_l2_error:.6f}"
        )
    print("  Layout performance is configuration-dependent.")


def run_method_comparison(identity_alpha: float) -> None:
    """Compare identity and smoothness assumptions on shared data."""
    dataframe, _ = run_regularization_comparison(
        {
            "identity": identity_alpha,
            "smoothness": SMOOTHNESS_ALPHA,
        },
        grid_shape=GRID_SHAPE,
        source_type=SOURCE_TYPE,
        sensor_strategy=SENSOR_STRATEGY,
        num_sensors=NUM_SENSORS,
        noise_level=NOISE_LEVEL,
        seed=MASTER_SEED,
    )
    dataframe.to_csv(
        OUTPUT_DIRECTORY / "regularization_method_comparison.csv",
        index=False,
    )

    print("\nIdentity-versus-smoothness comparison:")
    for row in dataframe.itertuples(index=False):
        print(
            f"  method={row.regularization}, alpha={row.alpha:.3e}, "
            f"relative L2 source error={row.relative_l2_error:.6f}, "
            f"residual norm={row.residual_norm:.6e}"
        )
    print(
        "  Smoothness regularization introduces a neighbour-smoothness "
        "modelling assumption."
    )
    print(
        "  The identity alpha was selected from the demonstrated identity "
        "regularization study."
    )
    print(
        "  The smoothness alpha is a fixed method-specific comparison value "
        "based on the existing Phase 3 tested scale."
    )
    print(
        "  The methods use differently scaled penalty operators, so their "
        "alpha values are not directly comparable."
    )
    print(
        "  No method is universally superior."
    )


def save_observation_diagnostics(result: ExperimentResult) -> None:
    """Analyze and save diagnostics for the selected benchmark matrix."""
    observation_matrix = build_observation_matrix(
        result.sensor_data_noisy.indices,
        result.grid,
    )
    diagnostics = analyze_observation_matrix(observation_matrix)
    json_diagnostics = to_json_safe(diagnostics)

    with (
        OUTPUT_DIRECTORY / "observation_diagnostics.json"
    ).open("w", encoding="utf-8") as output_file:
        json.dump(
            json_diagnostics,
            output_file,
            indent=2,
            allow_nan=False,
        )
        output_file.write("\n")

    figure, _ = plot_singular_values(
        diagnostics["singular_values"],
        rank_tolerance=diagnostics["rank_tolerance"],
        numerical_rank=diagnostics["numerical_rank"],
    )
    save_figure(
        figure,
        OUTPUT_DIRECTORY / "singular_values.png",
    )

    print("\nObservation-matrix diagnostics:")
    print(
        "  measurements="
        f"{diagnostics['number_of_measurements']}"
    )
    print(f"  unknowns={diagnostics['number_of_unknowns']}")
    print(
        "  measurement-to-unknown ratio="
        f"{diagnostics['measurement_to_unknown_ratio']:.6f}"
    )
    print(f"  numerical rank={diagnostics['numerical_rank']}")
    print(f"  nullity={diagnostics['nullity']}")
    print(f"  underdetermined={diagnostics['underdetermined']}")


def print_scientific_context() -> None:
    """Print concise interpretation limits for the synthetic studies."""
    print("\nScientific context:")
    print(
        "  Synthetic data supplies source-error metrics for validation; "
        "the package evaluates source errors on interior source nodes."
    )
    print(
        "  Residuals are measured in sensor space, and a low residual "
        "does not guarantee accurate source recovery."
    )
    print(
        "  Sparse reconstruction is underdetermined; sensor geometry "
        "and regularization assumptions both matter."
    )


def main() -> None:
    """Run every required study and save its documented outputs."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print_reference_configuration()

    identity_alpha, selected_result = run_identity_alpha_study()

    # This tested value belongs to the demonstrated reference setup. It
    # is held fixed below rather than re-optimized for every later study.
    run_sensor_count_example(identity_alpha)
    run_repeated_noise_example(identity_alpha)
    run_sensor_layout_example(identity_alpha)
    run_method_comparison(identity_alpha)
    save_observation_diagnostics(selected_result)
    print_scientific_context()

    print(f"\nOutputs: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
