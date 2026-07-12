"""Physical domain definitions for ThermoReconLab."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from thermoreconlab.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class Domain2D:
    """Represent a rectangular two-dimensional physical domain.

    The domain starts at ``(0, 0)`` and extends to
    ``(length_x, length_y)``.

    Parameters
    ----------
    length_x:
        Domain length in the x-direction.
    length_y:
        Domain length in the y-direction.

    Raises
    ------
    ValidationError
        If either length is not a finite positive number.
    """

    length_x: float = 1.0
    length_y: float = 1.0

    def __post_init__(self) -> None:
        """Validate and store the domain lengths as floats."""
        object.__setattr__(
            self,
            "length_x",
            self._validate_length(self.length_x, "length_x"),
        )
        object.__setattr__(
            self,
            "length_y",
            self._validate_length(self.length_y, "length_y"),
        )

    @staticmethod
    def _validate_length(value: float, name: str) -> float:
        """Validate one domain length."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValidationError(f"{name} must be a real number.")

        value = float(value)

        if not isfinite(value):
            raise ValidationError(f"{name} must be finite.")

        if value <= 0.0:
            raise ValidationError(f"{name} must be greater than zero.")

        return value

    @property
    def size(self) -> tuple[float, float]:
        """Return the domain lengths as ``(length_x, length_y)``."""
        return self.length_x, self.length_y

    @property
    def area(self) -> float:
        """Return the physical area of the domain."""
        return self.length_x * self.length_y

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return plotting bounds as ``(xmin, xmax, ymin, ymax)``."""
        return 0.0, self.length_x, 0.0, self.length_y