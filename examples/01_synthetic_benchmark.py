"""Run and export a complete ThermoReconLab synthetic benchmark.

This example demonstrates the public package workflow:

1. generate a known synthetic heat source,
2. solve the forward heat problem,
3. sample sparse noisy temperature measurements,
4. reconstruct the source,
5. compute validation metrics,
6. export figures, tables, JSON, and a Markdown report.
"""

from pathlib import Path

from thermoreconlab import run_synthetic_benchmark
from thermoreconlab.reporting import export_results


def main() -> None:
    """Run one reproducible benchmark and export its results."""
    result = run_synthetic_benchmark(
        grid_shape=(20, 20),
        source_type="two_gaussians",
        sensor_strategy="regular",
        num_sensors=16,
        noise_level=0.02,
        alpha=1e-7,
        seed=42,
    )

    output_directory = Path("outputs") / "synthetic_benchmark"

    exported_files = export_results(
        result,
        output_directory,
        dpi=160,
    )

    print("Synthetic benchmark completed.")
    print(f"Output directory: {exported_files['output_dir']}")
    print()
    print("Validation metrics:")

    for name, value in result.metrics.items():
        readable_name = name.replace("_", " ").title()
        print(f"  {readable_name}: {value:.6e}")

    print()
    print("Generated files:")
    print(f"  Metrics: {exported_files['metrics']}")
    print(f"  Summary: {exported_files['summary']}")
    print(f"  Report:  {exported_files['report']}")

    for name, path in exported_files["figures"].items():
        print(f"  Figure ({name}): {path}")


if __name__ == "__main__":
    main()

