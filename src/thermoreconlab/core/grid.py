"""Structured two-dimensional grids for ThermoReconLab."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from thermoreconlab.core.domain import Domain2D
from thermoreconlab.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class Grid2D:
    """Represent a structured two-dimensional Cartesian grid.

    The grid includes the boundary points of the physical domain.

    Parameters
    ----------
    nx:
        Number of grid points in the x-direction. Must be at least 3.
    ny:
        Number of grid points in the y-direction. Must be at least 3.
    domain:
        Physical domain covered by the grid.

    Raises
    ------
    ValidationError
        If the grid dimensions are invalid or the domain is not a
        :class:`Domain2D` object.
    """

    nx: int = 50
    ny: int = 50
    domain: Domain2D = field(default_factory=Domain2D)

    dx: float = field(init=False)
    dy: float = field(init=False)

    x: NDArray[np.float64] = field(init=False, repr=False)
    y: NDArray[np.float64] = field(init=False, repr=False)
    X: NDArray[np.float64] = field(init=False, repr=False)
    Y: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the grid and calculate its coordinate arrays."""
        nx = self._validate_point_count(self.nx, "nx")
        ny = self._validate_point_count(self.ny, "ny")

        if not isinstance(self.domain, Domain2D):
            raise ValidationError("domain must be a Domain2D object.")

        x = np.linspace(
            0.0,
            self.domain.length_x,
            nx,
            dtype=float,
        )
        y = np.linspace(
            0.0,
            self.domain.length_y,
            ny,
            dtype=float,
        )

        # indexing="ij" keeps all field arrays in shape (nx, ny).
        X, Y = np.meshgrid(x, y, indexing="ij")

        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "dx", self.domain.length_x / (nx - 1))
        object.__setattr__(self, "dy", self.domain.length_y / (ny - 1))
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Y", Y)

    @staticmethod
    def _validate_point_count(value: int, name: str) -> int:
        """Validate the number of grid points in one direction."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValidationError(f"{name} must be an integer.")

        value = int(value)

        if value < 3:
            raise ValidationError(f"{name} must be at least 3.")

        return value

    @property
    def shape(self) -> tuple[int, int]:
        """Return the field shape as ``(nx, ny)``."""
        return self.nx, self.ny

    @property
    def size(self) -> int:
        """Return the total number of grid points."""
        return self.nx * self.ny

    def mesh(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the two-dimensional coordinate arrays."""
        return self.X, self.Y

    def flatten(
        self,
        field_array: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Flatten a two-dimensional field using C-order indexing.

        Parameters
        ----------
        field_array:
            Two-dimensional field with shape equal to ``grid.shape``.

        Returns
        -------
        numpy.ndarray
            One-dimensional copy with length ``grid.size``.
        """
        array = np.asarray(field_array, dtype=float)

        if array.shape != self.shape:
            raise ValidationError(
                f"field_array must have shape {self.shape}, "
                f"but received {array.shape}."
            )

        if not np.all(np.isfinite(array)):
            raise ValidationError(
                "field_array must contain only finite values."
            )

        return array.reshape(self.size, order="C").copy()

    def reshape(
        self,
        vector: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Reshape a one-dimensional vector into the grid shape.

        Parameters
        ----------
        vector:
            One-dimensional vector with length equal to ``grid.size``.

        Returns
        -------
        numpy.ndarray
            Two-dimensional copy with shape ``grid.shape``.
        """
        array = np.asarray(vector, dtype=float)

        if array.ndim != 1:
            raise ValidationError("vector must be one-dimensional.")

        if array.size != self.size:
            raise ValidationError(
                f"vector must contain {self.size} values, "
                f"but received {array.size}."
            )

        if not np.all(np.isfinite(array)):
            raise ValidationError("vector must contain only finite values.")

        return array.reshape(self.shape, order="C").copy()