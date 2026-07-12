"""Numerical core components for ThermoReconLab."""

from thermoreconlab.core.domain import Domain2D
from thermoreconlab.core.fields import (
    ensure_2d_array,
    flatten_field,
    reshape_field,
    validate_field,
)
from thermoreconlab.core.grid import Grid2D

__all__ = [
    "Domain2D",
    "Grid2D",
    "ensure_2d_array",
    "validate_field",
    "flatten_field",
    "reshape_field",
]