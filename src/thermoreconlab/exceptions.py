"""Custom exceptions used throughout ThermoReconLab."""


class ThermoReconLabError(Exception):
    """Base exception for all package-specific errors."""


class ValidationError(ThermoReconLabError):
    """Raised when an input value or array fails validation."""


class DataFormatError(ThermoReconLabError):
    """Raised when imported data has an unsupported or invalid format."""


class SolverError(ThermoReconLabError):
    """Raised when a numerical solver cannot produce a valid result."""