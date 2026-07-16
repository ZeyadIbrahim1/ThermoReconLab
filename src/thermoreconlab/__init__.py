"""ThermoReconLab package.

ThermoReconLab provides tools for reproducible two-dimensional
heat-source reconstruction experiments using sparse temperature
sensor measurements.
"""

from thermoreconlab.reconstruction import (
    reconstruct_compact_nonnegative,
    reconstruct_smooth_tikhonov,
)
from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
    reconstruct_from_measurements,
    run_compact_parameter_study,
    run_reconstruction_method_study,
    run_synthetic_benchmark,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ExperimentResult",
    "MeasurementReconstructionResult",
    "run_compact_parameter_study",
    "run_reconstruction_method_study",
    "run_synthetic_benchmark",
    "reconstruct_from_measurements",
    "reconstruct_compact_nonnegative",
    "reconstruct_smooth_tikhonov",
]
