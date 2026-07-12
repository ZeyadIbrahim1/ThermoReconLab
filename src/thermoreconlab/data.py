"""Synthetic source generation and array input/output.

This module supports the two data-entry modes of ThermoReconLab:

1. Synthetic source fields for validation and benchmarking.
2. User-provided two-dimensional arrays loaded from files.
"""

from __future__ import annotations

from numbers import Integral, Real
from os import PathLike
from pathlib import Path
import pandas as pd

from thermoreconlab.sensors import SensorData
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from thermoreconlab.core.fields import ensure_2d_array, validate_field
from thermoreconlab.core.grid import Grid2D
from thermoreconlab.exceptions import DataFormatError, ValidationError


PathInput = str | PathLike[str]
Center = tuple[float, float]


def _validate_grid(grid: Grid2D) -> None:
    """Check that an object is a valid two-dimensional grid."""
    if not isinstance(grid, Grid2D):
        raise ValidationError("grid must be a Grid2D object.")


def _validate_real(
    value: Real,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Validate and convert a real-valued parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number.")

    result = float(value)

    if not np.isfinite(result):
        raise ValidationError(f"{name} must be finite.")

    if positive and result <= 0.0:
        raise ValidationError(f"{name} must be greater than zero.")

    if nonnegative and result < 0.0:
        raise ValidationError(f"{name} must be nonnegative.")

    return result


def _validate_center(
    center: Sequence[Real],
    grid: Grid2D,
    *,
    name: str = "center",
) -> Center:
    """Validate a physical source center inside the domain."""
    if isinstance(center, (str, bytes)) or len(center) != 2:
        raise ValidationError(
            f"{name} must contain exactly two coordinates."
        )

    center_x = _validate_real(center[0], f"{name}[0]")
    center_y = _validate_real(center[1], f"{name}[1]")

    if not 0.0 <= center_x <= grid.domain.length_x:
        raise ValidationError(
            f"{name}[0] must lie between 0 and "
            f"{grid.domain.length_x}."
        )

    if not 0.0 <= center_y <= grid.domain.length_y:
        raise ValidationError(
            f"{name}[1] must lie between 0 and "
            f"{grid.domain.length_y}."
        )

    return center_x, center_y


def gaussian_source(
    grid: Grid2D,
    center: Sequence[Real] = (0.5, 0.5),
    amplitude: Real = 1.0,
    sigma: Real = 0.08,
) -> NDArray[np.float64]:
    """Generate a two-dimensional Gaussian heat-source field.

    Parameters
    ----------
    grid:
        Grid on which the source is evaluated.
    center:
        Physical ``(x, y)`` coordinates of the Gaussian center.
    amplitude:
        Peak source intensity.
    sigma:
        Gaussian width in physical domain units.

    Returns
    -------
    numpy.ndarray
        Nonnegative heat-source field with shape ``grid.shape``.
    """
    _validate_grid(grid)

    center_x, center_y = _validate_center(center, grid)
    amplitude_value = _validate_real(
        amplitude,
        "amplitude",
        nonnegative=True,
    )
    sigma_value = _validate_real(
        sigma,
        "sigma",
        positive=True,
    )

    squared_distance = (
        (grid.X - center_x) ** 2
        + (grid.Y - center_y) ** 2
    )

    source = amplitude_value * np.exp(
        -squared_distance / (2.0 * sigma_value**2)
    )

    return source.astype(float, copy=False)


def two_gaussian_sources(
    grid: Grid2D,
    centers: Sequence[Sequence[Real]] = (
        (0.35, 0.40),
        (0.70, 0.65),
    ),
    amplitudes: Sequence[Real] = (1.0, 0.7),
    sigmas: Sequence[Real] = (0.07, 0.09),
) -> NDArray[np.float64]:
    """Generate a source field containing two Gaussian hotspots.

    Parameters
    ----------
    grid:
        Grid on which the source is evaluated.
    centers:
        Physical coordinates of the two source centers.
    amplitudes:
        Peak intensities of the two sources.
    sigmas:
        Widths of the two Gaussian sources.

    Returns
    -------
    numpy.ndarray
        Combined nonnegative source field.
    """
    _validate_grid(grid)

    if len(centers) != 2:
        raise ValidationError("centers must contain exactly two centers.")

    if len(amplitudes) != 2:
        raise ValidationError(
            "amplitudes must contain exactly two values."
        )

    if len(sigmas) != 2:
        raise ValidationError(
            "sigmas must contain exactly two values."
        )

    first_source = gaussian_source(
        grid,
        center=centers[0],
        amplitude=amplitudes[0],
        sigma=sigmas[0],
    )

    second_source = gaussian_source(
        grid,
        center=centers[1],
        amplitude=amplitudes[1],
        sigma=sigmas[1],
    )

    return first_source + second_source


def random_hotspots(
    grid: Grid2D,
    n_hotspots: int = 3,
    *,
    seed: int | None = None,
    amplitude_range: tuple[Real, Real] = (0.5, 1.0),
    sigma_range: tuple[Real, Real] = (0.04, 0.10),
) -> NDArray[np.float64]:
    """Generate reproducible randomly positioned Gaussian hotspots.

    Parameters
    ----------
    grid:
        Grid on which the source is evaluated.
    n_hotspots:
        Number of Gaussian hotspots.
    seed:
        Optional random seed for reproducibility.
    amplitude_range:
        Minimum and maximum random source amplitudes.
    sigma_range:
        Minimum and maximum random source widths.

    Returns
    -------
    numpy.ndarray
        Combined random source field.
    """
    _validate_grid(grid)

    if (
        isinstance(n_hotspots, bool)
        or not isinstance(n_hotspots, Integral)
    ):
        raise ValidationError("n_hotspots must be an integer.")

    n_hotspots = int(n_hotspots)

    if n_hotspots <= 0:
        raise ValidationError(
            "n_hotspots must be greater than zero."
        )

    amplitude_min = _validate_real(
        amplitude_range[0],
        "amplitude_range[0]",
        nonnegative=True,
    )
    amplitude_max = _validate_real(
        amplitude_range[1],
        "amplitude_range[1]",
        nonnegative=True,
    )

    sigma_min = _validate_real(
        sigma_range[0],
        "sigma_range[0]",
        positive=True,
    )
    sigma_max = _validate_real(
        sigma_range[1],
        "sigma_range[1]",
        positive=True,
    )

    if amplitude_min > amplitude_max:
        raise ValidationError(
            "amplitude_range minimum must not exceed its maximum."
        )

    if sigma_min > sigma_max:
        raise ValidationError(
            "sigma_range minimum must not exceed its maximum."
        )

    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "seed must be an integer or None."
        ) from error

    source = np.zeros(grid.shape, dtype=float)

    for _ in range(n_hotspots):
        center = (
            rng.uniform(0.0, grid.domain.length_x),
            rng.uniform(0.0, grid.domain.length_y),
        )
        amplitude = rng.uniform(amplitude_min, amplitude_max)
        sigma = rng.uniform(sigma_min, sigma_max)

        source += gaussian_source(
            grid,
            center=center,
            amplitude=amplitude,
            sigma=sigma,
        )

    return source


def custom_source(
    array: ArrayLike,
    grid: Grid2D,
    *,
    require_nonnegative: bool = True,
) -> NDArray[np.float64]:
    """Validate a user-defined heat-source array.

    Parameters
    ----------
    array:
        User-provided two-dimensional source field.
    grid:
        Grid defining the required array shape.
    require_nonnegative:
        Whether negative source values should be rejected.

    Returns
    -------
    numpy.ndarray
        Validated copy of the source field.
    """
    source = validate_field(array, grid, name="source")

    if require_nonnegative and np.any(source < 0.0):
        raise ValidationError(
            "source must contain only nonnegative values."
        )

    return source


def load_array(
    path: PathInput,
    *,
    array_name: str | None = None,
) -> NDArray[np.float64]:
    """Load a two-dimensional array from CSV, TXT, NPY, or NPZ.

    Parameters
    ----------
    path:
        Input file path.
    array_name:
        Name of the array to load from an NPZ archive. It is only
        required when an archive contains multiple arrays and no
        array named ``array``.

    Returns
    -------
    numpy.ndarray
        Validated two-dimensional floating-point array.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    DataFormatError
        If the extension is unsupported or the file cannot be read.
    ValidationError
        If the loaded data is not a finite two-dimensional array.
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Data file does not exist: {file_path}"
        )

    extension = file_path.suffix.lower()

    try:
        if extension == ".csv":
            array = np.loadtxt(file_path, delimiter=",")

        elif extension == ".txt":
            array = np.loadtxt(file_path)

        elif extension == ".npy":
            array = np.load(file_path, allow_pickle=False)

        elif extension == ".npz":
            with np.load(file_path, allow_pickle=False) as archive:
                available_names = list(archive.files)

                if array_name is not None:
                    if array_name not in available_names:
                        raise DataFormatError(
                            f"Array '{array_name}' was not found in "
                            f"{file_path.name}. Available arrays: "
                            f"{available_names}"
                        )
                    selected_name = array_name

                elif "array" in available_names:
                    selected_name = "array"

                elif len(available_names) == 1:
                    selected_name = available_names[0]

                else:
                    raise DataFormatError(
                        f"{file_path.name} contains multiple arrays. "
                        "Specify array_name."
                    )

                array = archive[selected_name]

        else:
            raise DataFormatError(
                f"Unsupported file extension '{extension}'. "
                "Supported extensions are .csv, .txt, .npy, and .npz."
            )

    except DataFormatError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise DataFormatError(
            f"Could not read data from {file_path}."
        ) from error

    return ensure_2d_array(
        array,
        name=f"data loaded from '{file_path.name}'",
    )


def save_array(
    path: PathInput,
    array: ArrayLike,
) -> Path:
    """Save a two-dimensional array as CSV, TXT, NPY, or NPZ.

    Parent directories are created automatically when necessary.

    Parameters
    ----------
    path:
        Output file path.
    array:
        Two-dimensional numeric array to save.

    Returns
    -------
    pathlib.Path
        Final output path.
    """
    file_path = Path(path)
    clean_array = ensure_2d_array(array)

    extension = file_path.suffix.lower()

    if extension not in {".csv", ".txt", ".npy", ".npz"}:
        raise DataFormatError(
            f"Unsupported file extension '{extension}'. "
            "Supported extensions are .csv, .txt, .npy, and .npz."
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if extension == ".csv":
            np.savetxt(file_path, clean_array, delimiter=",")

        elif extension == ".txt":
            np.savetxt(file_path, clean_array)

        elif extension == ".npy":
            np.save(file_path, clean_array)

        else:
            np.savez_compressed(file_path, array=clean_array)

    except OSError as error:
        raise DataFormatError(
            f"Could not save data to {file_path}."
        ) from error

    return file_path


def load_heatmap_csv(
    path: PathInput,
    *,
    grid: Grid2D | None = None,
) -> NDArray[np.float64]:
    """Load a two-dimensional heatmap from a CSV file.

    When a grid is supplied, the loaded heatmap must match the grid
    shape.
    """
    file_path = Path(path)

    if file_path.suffix.lower() != ".csv":
        raise DataFormatError(
            "load_heatmap_csv requires a .csv file."
        )

    array = load_array(file_path)

    if grid is not None:
        return validate_field(array, grid, name="heatmap")

    return array
def load_sensor_csv(path: PathInput) -> SensorData:
    """Load sparse sensor measurements from a CSV file.

    The CSV file must contain these columns:

    - ``i``: grid index in the x-direction,
    - ``j``: grid index in the y-direction,
    - ``value``: measured temperature.

    Optional physical-coordinate columns are:

    - ``x``,
    - ``y``.

    Parameters
    ----------
    path:
        Path to the sensor CSV file.

    Returns
    -------
    SensorData
        Validated sensor locations and measurements.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    DataFormatError
        If the file is not CSV or cannot be read.
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Sensor data file does not exist: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise DataFormatError(
            "Sensor measurements must be loaded from a CSV file."
        )

    try:
        dataframe = pd.read_csv(file_path)
    except (OSError, pd.errors.ParserError) as error:
        raise DataFormatError(
            f"Could not read sensor data from {file_path}."
        ) from error

    try:
        return SensorData.from_dataframe(dataframe)
    except ValidationError as error:
        raise DataFormatError(
            f"Invalid sensor CSV format in {file_path.name}. "
            "Required columns are i, j, and value."
        ) from error


def save_sensor_csv(
    path: PathInput,
    sensor_data: SensorData,
) -> Path:
    """Save sparse sensor measurements to a CSV file.

    Parameters
    ----------
    path:
        Output CSV path.
    sensor_data:
        Sensor measurements to save.

    Returns
    -------
    pathlib.Path
        Final output path.

    Raises
    ------
    ValidationError
        If ``sensor_data`` is invalid.
    DataFormatError
        If the path is not a CSV file or writing fails.
    """
    if not isinstance(sensor_data, SensorData):
        raise ValidationError(
            "sensor_data must be a SensorData object."
        )

    file_path = Path(path)

    if file_path.suffix.lower() != ".csv":
        raise DataFormatError(
            "Sensor measurements must be saved as a CSV file."
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sensor_data.to_dataframe().to_csv(
            file_path,
            index=False,
        )
    except OSError as error:
        raise DataFormatError(
            f"Could not save sensor data to {file_path}."
        ) from error

    return file_path