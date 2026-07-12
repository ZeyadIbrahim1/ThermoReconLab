"""Tests for the two-dimensional domain definition."""

import pytest

from thermoreconlab.core.domain import Domain2D
from thermoreconlab.exceptions import ValidationError


def test_default_domain() -> None:
    """The default domain should be the unit square."""
    domain = Domain2D()

    assert domain.length_x == 1.0
    assert domain.length_y == 1.0
    assert domain.size == (1.0, 1.0)
    assert domain.area == 1.0
    assert domain.extent == (0.0, 1.0, 0.0, 1.0)


def test_rectangular_domain() -> None:
    """The domain should support different x and y lengths."""
    domain = Domain2D(length_x=2.0, length_y=3.0)

    assert domain.size == (2.0, 3.0)
    assert domain.area == 6.0
    assert domain.extent == (0.0, 2.0, 0.0, 3.0)


def test_integer_lengths_are_converted_to_float() -> None:
    """Integer inputs should be accepted and stored as floats."""
    domain = Domain2D(length_x=2, length_y=4)

    assert domain.length_x == 2.0
    assert domain.length_y == 4.0
    assert isinstance(domain.length_x, float)
    assert isinstance(domain.length_y, float)


@pytest.mark.parametrize(
    "length_x, length_y",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
    ],
)
def test_non_positive_lengths_raise_error(
    length_x: float,
    length_y: float,
) -> None:
    """Zero and negative domain lengths should be rejected."""
    with pytest.raises(ValidationError):
        Domain2D(length_x=length_x, length_y=length_y)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_lengths_raise_error(invalid_value: float) -> None:
    """NaN and infinite lengths should be rejected."""
    with pytest.raises(ValidationError):
        Domain2D(length_x=invalid_value)


@pytest.mark.parametrize("invalid_value", ["1.0", None, True])
def test_non_numeric_lengths_raise_error(invalid_value: object) -> None:
    """Non-numeric values and Boolean values should be rejected."""
    with pytest.raises(ValidationError):
        Domain2D(length_x=invalid_value)  # type: ignore[arg-type]