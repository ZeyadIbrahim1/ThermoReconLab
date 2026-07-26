"""ThermoReconLab package.

ThermoReconLab provides tools for reproducible two-dimensional
heat-source reconstruction from sparse sensor measurements or supplied
temperature fields.
"""

from thermoreconlab.experiments import (
    ExperimentResult,
    MeasurementReconstructionResult,
    reconstruct_from_measurements,
    reconstruct_from_temperature_field,
    run_compact_parameter_study,
    run_reconstruction_method_study,
    run_regularization_comparison,
    run_synthetic_benchmark,
)
from thermoreconlab.reconstruction import (
    reconstruct_compact_nonnegative,
    reconstruct_smooth_tikhonov,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ExperimentResult",
    "MeasurementReconstructionResult",
    "reconstruct_compact_nonnegative",
    "reconstruct_from_measurements",
    "reconstruct_from_temperature_field",
    "reconstruct_smooth_tikhonov",
    "run_compact_parameter_study",
    "run_reconstruction_method_study",
    "run_regularization_comparison",
    "run_synthetic_benchmark",
]
