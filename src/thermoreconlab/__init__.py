"""ThermoReconLab package.

ThermoReconLab provides tools for reproducible two-dimensional
heat-source reconstruction experiments using sparse temperature
sensor measurements.
"""

from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
    reconstruct_from_measurements,
    run_synthetic_benchmark,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ExperimentResult",
    "MeasurementReconstructionResult",
    "run_synthetic_benchmark",
    "reconstruct_from_measurements",
]