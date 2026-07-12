"""Tests for synthetic data generation and array input/output."""

from pathlib import Path
import pandas as pd

from thermoreconlab.sensors import SensorData
import numpy as np
import pytest

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.data import (
    custom_source,
    gaussian_source,
    load_array,
    load_heatmap_csv,
    load_sensor_csv,
    random_hotspots,
    save_array,
    save_sensor_csv,
    two_gaussian_sources,
)
from thermoreconlab.exceptions import DataFormatError, ValidationError


def test_gaussian_source_has_correct_shape() -> None:
    """A Gaussian source should match the grid shape."""
    grid = Grid2D(nx=21, ny=25)

    source = gaussian_source(grid)

    assert source.shape == grid.shape
    assert source.dtype == np.float64
    assert np.all(np.isfinite(source))
    assert np.all(source >= 0.0)


def test_gaussian_source_peak_matches_amplitude() -> None:
    """A centered Gaussian should reach its amplitude on the grid."""
    grid = Grid2D(nx=5, ny=5)

    source = gaussian_source(
        grid,
        center=(0.5, 0.5),
        amplitude=2.0,
        sigma=0.1,
    )

    assert source[2, 2] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "sigma",
    [0.0, -0.1],
)
def test_gaussian_source_rejects_invalid_sigma(
    sigma: float,
) -> None:
    """Gaussian width must be positive."""
    grid = Grid2D(nx=5, ny=5)

    with pytest.raises(ValidationError):
        gaussian_source(grid, sigma=sigma)


def test_gaussian_source_rejects_negative_amplitude() -> None:
    """Synthetic heat-source amplitudes should be nonnegative."""
    grid = Grid2D(nx=5, ny=5)

    with pytest.raises(ValidationError):
        gaussian_source(grid, amplitude=-1.0)


@pytest.mark.parametrize(
    "center",
    [
        (-0.1, 0.5),
        (1.1, 0.5),
        (0.5, -0.1),
        (0.5, 1.1),
    ],
)
def test_gaussian_source_rejects_center_outside_domain(
    center: tuple[float, float],
) -> None:
    """Source centers must lie inside the physical domain."""
    grid = Grid2D(nx=5, ny=5)

    with pytest.raises(ValidationError):
        gaussian_source(grid, center=center)


def test_two_gaussian_sources_are_nonnegative() -> None:
    """The combined two-source benchmark should remain valid."""
    grid = Grid2D(nx=20, ny=20)

    source = two_gaussian_sources(grid)

    assert source.shape == grid.shape
    assert np.all(source >= 0.0)
    assert np.max(source) > 0.0


def test_random_hotspots_are_reproducible() -> None:
    """The same random seed should produce the same field."""
    grid = Grid2D(nx=20, ny=20)

    first = random_hotspots(grid, n_hotspots=4, seed=42)
    second = random_hotspots(grid, n_hotspots=4, seed=42)

    assert np.array_equal(first, second)


def test_random_hotspots_change_with_seed() -> None:
    """Different seeds should normally produce different fields."""
    grid = Grid2D(nx=20, ny=20)

    first = random_hotspots(grid, seed=1)
    second = random_hotspots(grid, seed=2)

    assert not np.array_equal(first, second)


def test_random_hotspots_rejects_invalid_count() -> None:
    """At least one hotspot is required."""
    grid = Grid2D(nx=10, ny=10)

    with pytest.raises(ValidationError):
        random_hotspots(grid, n_hotspots=0)


def test_custom_source_accepts_matching_array() -> None:
    """A valid custom source should be copied and returned."""
    grid = Grid2D(nx=4, ny=5)
    source = np.ones(grid.shape)

    result = custom_source(source, grid)
    result[0, 0] = 3.0

    assert result.shape == grid.shape
    assert source[0, 0] == 1.0


def test_custom_source_rejects_negative_values() -> None:
    """Negative values should be rejected by default."""
    grid = Grid2D(nx=4, ny=5)
    source = np.ones(grid.shape)
    source[2, 2] = -1.0

    with pytest.raises(ValidationError):
        custom_source(source, grid)


@pytest.mark.parametrize(
    "extension",
    [".csv", ".txt", ".npy", ".npz"],
)
def test_array_save_and_load_round_trip(
    tmp_path: Path,
    extension: str,
) -> None:
    """Supported formats should preserve a two-dimensional array."""
    array = np.arange(20, dtype=float).reshape(4, 5)
    path = tmp_path / f"field{extension}"

    saved_path = save_array(path, array)
    loaded = load_array(saved_path)

    assert saved_path.exists()
    assert loaded.shape == array.shape
    assert np.allclose(loaded, array)


def test_save_array_creates_parent_directory(
    tmp_path: Path,
) -> None:
    """Missing output folders should be created automatically."""
    array = np.ones((3, 4))
    path = tmp_path / "nested" / "folder" / "field.npy"

    saved_path = save_array(path, array)

    assert saved_path.exists()


def test_load_array_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing input file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_array(tmp_path / "missing.csv")


def test_unsupported_extension_raises_error(
    tmp_path: Path,
) -> None:
    """Unsupported file formats should produce a clear error."""
    path = tmp_path / "field.json"
    path.write_text("[[1, 2], [3, 4]]", encoding="utf-8")

    with pytest.raises(DataFormatError):
        load_array(path)


def test_npz_with_multiple_arrays_requires_name(
    tmp_path: Path,
) -> None:
    """Ambiguous NPZ archives should require an array name."""
    path = tmp_path / "multiple.npz"

    np.savez(
        path,
        first=np.ones((2, 2)),
        second=np.zeros((2, 2)),
    )

    with pytest.raises(DataFormatError):
        load_array(path)

    selected = load_array(path, array_name="second")

    assert np.array_equal(selected, np.zeros((2, 2)))


def test_load_heatmap_csv_checks_grid_shape(
    tmp_path: Path,
) -> None:
    """CSV heatmaps should optionally be checked against a grid."""
    path = tmp_path / "heatmap.csv"
    array = np.ones((4, 5))
    save_array(path, array)

    grid = Grid2D(nx=4, ny=5)
    loaded = load_heatmap_csv(path, grid=grid)

    assert loaded.shape == grid.shape


def test_load_heatmap_csv_rejects_wrong_shape(
    tmp_path: Path,
) -> None:
    """A heatmap inconsistent with the grid should be rejected."""
    path = tmp_path / "heatmap.csv"
    save_array(path, np.ones((5, 4)))

    grid = Grid2D(nx=4, ny=5)

    with pytest.raises(ValidationError):
        load_heatmap_csv(path, grid=grid)

def test_sensor_csv_round_trip(tmp_path: Path) -> None:
    """Sensor measurements should survive CSV save and load."""
    sensor_data = SensorData(
        indices=np.array(
            [
                [1, 2],
                [3, 4],
            ]
        ),
        values=np.array([0.5, 1.5]),
        coordinates=np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        ),
    )

    path = tmp_path / "sensors.csv"

    saved_path = save_sensor_csv(path, sensor_data)
    loaded = load_sensor_csv(saved_path)

    assert saved_path.exists()
    assert np.array_equal(
        loaded.indices,
        sensor_data.indices,
    )
    assert np.allclose(
        loaded.values,
        sensor_data.values,
    )
    assert loaded.coordinates is not None
    assert np.allclose(
        loaded.coordinates,
        sensor_data.coordinates,
    )


def test_load_sensor_csv_accepts_required_columns(
    tmp_path: Path,
) -> None:
    """A CSV containing i, j, and value should be valid."""
    path = tmp_path / "sensors.csv"

    dataframe = pd.DataFrame(
        {
            "i": [1, 2],
            "j": [3, 4],
            "value": [0.5, 0.8],
        }
    )
    dataframe.to_csv(path, index=False)

    sensor_data = load_sensor_csv(path)

    assert len(sensor_data) == 2
    assert sensor_data.coordinates is None


def test_load_sensor_csv_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """The required sensor columns must be present."""
    path = tmp_path / "invalid_sensors.csv"

    dataframe = pd.DataFrame(
        {
            "i": [1, 2],
            "value": [0.5, 0.8],
        }
    )
    dataframe.to_csv(path, index=False)

    with pytest.raises(DataFormatError):
        load_sensor_csv(path)


def test_load_sensor_csv_rejects_non_csv_file(
    tmp_path: Path,
) -> None:
    """Sensor measurements currently require CSV input."""
    path = tmp_path / "sensors.txt"
    path.write_text("i,j,value\n1,2,0.5", encoding="utf-8")

    with pytest.raises(DataFormatError):
        load_sensor_csv(path)


def test_save_sensor_csv_rejects_invalid_object(
    tmp_path: Path,
) -> None:
    """Only SensorData objects should be exported."""
    with pytest.raises(ValidationError):
        save_sensor_csv(
            tmp_path / "sensors.csv",
            np.ones((3, 3)),  # type: ignore[arg-type]
        )

