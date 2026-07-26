"""Demonstrate ThermoReconLab user-measurement reconstruction.

This example creates a small demonstration sensor CSV, reloads it
through the package data interface, reconstructs a heat-source field,
and exports the user-mode result package.

The synthetic generation is used only to create reproducible example
measurements. The reconstruction itself uses the CSV file as external
input and does not access the true source.
"""

from pathlib import Path

from thermoreconlab import reconstruct_from_measurements
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.data import (
    gaussian_source,
    load_sensor_csv,
    save_sensor_csv,
)
from thermoreconlab.reconstruction import solve_forward
from thermoreconlab.reporting import export_results
from thermoreconlab.sensors import (
    create_sensor_data,
    regular_grid_sensors,
)


GRID_SHAPE = (20, 20)
NUM_SENSORS = 16
ALPHA = 1e-7


def create_demo_sensor_csv(csv_path: Path) -> None:
    """Create reproducible demonstration measurements."""
    grid = Grid2D(
        nx=GRID_SHAPE[0],
        ny=GRID_SHAPE[1],
    )

    source = gaussian_source(
        grid,
        center=(0.55, 0.45),
        amplitude=1.0,
        sigma=0.09,
    )

    temperature = solve_forward(source, grid)

    sensor_indices = regular_grid_sensors(
        grid,
        count=NUM_SENSORS,
        include_boundary=False,
    )

    sensor_data = create_sensor_data(
        temperature,
        sensor_indices,
        grid,
    )

    save_sensor_csv(csv_path, sensor_data)


def main() -> None:
    """Load sensor CSV data, reconstruct, and export results."""
    data_directory = Path("examples") / "data"
    data_directory.mkdir(parents=True, exist_ok=True)

    csv_path = data_directory / "demo_sensor_measurements.csv"

    if not csv_path.exists():
        create_demo_sensor_csv(csv_path)
        print(f"Created demonstration sensor file: {csv_path}")

    sensor_data = load_sensor_csv(csv_path)

    result = reconstruct_from_measurements(
        sensor_data,
        grid_shape=GRID_SHAPE,
        alpha=ALPHA,
    )

    output_directory = Path("outputs") / "user_measurement_example"

    exported_files = export_results(
        result,
        output_directory,
    )

    print("User-measurement reconstruction completed.")
    print(f"Sensor file: {csv_path}")
    print(f"Number of sensors: {len(sensor_data)}")
    print(
        "Residual norm: "
        f"{result.reconstruction.residual_norm:.6e}"
    )
    print(f"Output directory: {exported_files['output_dir']}")
    print(f"Report: {exported_files['report']}")


if __name__ == "__main__":
    main()
