"""Tests for the Phase 5 read-only HDF5 inspector."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "ai_data.py"
SPEC = importlib.util.spec_from_file_location("ai_data", MODULE_PATH)
ai_data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ai_data
SPEC.loader.exec_module(ai_data)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["project"] = "audit"
        group = handle.create_group("measurements")
        group.attrs["plate"] = "DF"
        temperature = group.create_dataset(
            "temperature_C", data=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            chunks=(1, 3, 4), compression="gzip",
        )
        temperature.attrs["units"] = "degC"
        group.create_dataset("heat_flux_kW_m2", data=np.ones((3, 4)))
        group.create_dataset("time_s", data=np.arange(2.0))
        group.create_dataset("x_coordinate_m", data=np.arange(4.0))
        group.create_dataset("y_coordinate_m", data=np.arange(3.0))
        handle.create_dataset("scalar", data=np.float64(3.5))
        handle.create_dataset("matrix", data=np.eye(3))
        handle.create_dataset("higher", shape=(2, 2, 2, 2), dtype="i2")
        handle.create_dataset("nonfinite", data=[0.0, np.nan, np.inf, -np.inf])
    return path


@pytest.fixture
def pipeline_config() -> dict:
    return {
        "schema_version": 2, "random_seed": 10, "output_directory": "unused",
        "grid_shape": [6, 6], "num_samples": 6,
        "source_family_probabilities": {
            "one_gaussian": 0.4, "multiple_gaussians": 0.2,
            "elliptical_hotspot": 0.1, "rectangle": 0.1,
            "circular_compact": 0.1, "elongated_source": 0.05,
            "overlapping_hotspots": 0.03, "irregular_composite": 0.02,
            "sharp_edged": 0.0,
        },
        "source_count_range": [2, 3], "allow_signed_sources": False,
        "signed_probability": 0.0, "amplitude_range": [0.8, 1.2],
        "width_range": [0.08, 0.12], "size_range": [0.1, 0.2],
        "sensor_strategies": ["regular_grid", "random", "center_focused"],
        "sensor_count_range": [4, 5], "sensor_seeds": [100],
        "noise_levels": [0.0, 0.01, 0.1],
        "identity_alpha_choices": [0.01], "smoothness_alpha_choices": [0.001],
        "split_rules": {"counts": {role: 1 for role in ai_data.SPLIT_ROLES}},
        "ood_source_families": ["sharp_edged"],
        "ood_sensor_strategies": ["center_focused"], "ood_noise_levels": [0.1],
        "normalization": {"enabled": True, "method": "global_standard"},
        "storage_compression": "gzip", "preview_count": 1,
    }


def make_external_pair(tmp_path: Path, frames: tuple[int, ...] = (0, 1, 2, 3, 4)) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    temperature = tmp_path / "T.h5"
    flux = tmp_path / "HF.h5"
    with h5py.File(temperature, "w") as t_handle, h5py.File(flux, "w") as f_handle:
        for frame in frames:
            values = np.full((6, 7), frame + 1.0)
            t_handle.create_dataset(f"surface_temperature_batch0_frame{frame:06d}", data=values)
            f_handle.create_dataset(f"estimated_flux_batch0_frame{frame:06d}", data=10.0 * values)
    return temperature, flux


def resign_manifest(directory: Path) -> None:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["dataset_sha256"] = digest(directory / "synthetic_dataset.h5")
    manifest["manifest_content_sha256"] = ai_data._manifest_content_hash(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def resign_external_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text())
    manifest["manifest_content_sha256"] = ai_data._manifest_content_hash(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_recursive_structure_metadata_and_semantics(sample_file: Path) -> None:
    report = ai_data.inspect_hdf5(sample_file)
    assert report["schema_version"] == 1
    assert report["open_mode"] == "read-only"
    assert report["root_attributes"] == {"project": "audit"}
    assert report["groups"] == [{"path": "/measurements", "attributes": {"plate": "DF"}}]
    by_path = {item["path"]: item for item in report["datasets"]}
    assert report["dataset_paths"] == sorted(report["dataset_paths"])
    temp = by_path["/measurements/temperature_C"]
    assert temp["shape"] == [2, 3, 4]
    assert temp["dtype"] == "float32"
    assert temp["chunks"] == [1, 3, 4]
    assert temp["compression"] == "gzip"
    assert temp["logical_size_bytes"] == 96
    assert temp["attributes"]["units"] == "degC"
    assert by_path["/scalar"]["shape"] == []
    assert by_path["/matrix"]["ndim"] == 2
    assert by_path["/higher"]["ndim"] == 4
    semantics = report["semantic_inference"]
    assert semantics["temperature"]["selected"] == "/measurements/temperature_C"
    assert semantics["heat_flux"]["selected"] == "/measurements/heat_flux_kW_m2"
    assert semantics["time"]["selected"] == "/measurements/time_s"
    assert semantics["x_coordinate"]["selected"] == "/measurements/x_coordinate_m"
    assert semantics["y_coordinate"]["selected"] == "/measurements/y_coordinate_m"
    assert semantics["temperature"]["alternatives"][0]["rule"]


def test_nonfinite_statistics(sample_file: Path) -> None:
    item = next(x for x in ai_data.inspect_hdf5(sample_file)["datasets"] if x["path"] == "/nonfinite")
    sample = item["sample"]
    assert sample["all_finite"] is False
    assert sample["finite_count"] == 1
    assert sample["nan_count"] == 1
    assert sample["positive_infinity_count"] == 1
    assert sample["negative_infinity_count"] == 1
    assert sample["minimum"] == sample["maximum"] == sample["mean"] == 0.0


@pytest.mark.parametrize("shape", [(), (8,), (3, 4), (2, 3, 4), (2, 2, 2, 2, 2)])
def test_dimensionality_and_bounded_sampling(tmp_path: Path, shape: tuple[int, ...]) -> None:
    path = tmp_path / "dimensions.h5"
    data = np.asarray(7) if not shape else np.arange(np.prod(shape)).reshape(shape)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("values", data=data)
    item = ai_data.inspect_hdf5(path, sample_limit=5)["datasets"][0]
    assert item["ndim"] == len(shape)
    assert item["shape"] == list(shape)
    assert item["sample"]["sampled_elements"] <= 5


def test_large_dataset_uses_bounded_hyperslab(tmp_path: Path) -> None:
    path = tmp_path / "large.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("large", shape=(10_000, 10_000), dtype="f8", chunks=(10, 10))
    item = ai_data.inspect_hdf5(path, sample_limit=17)["datasets"][0]
    assert item["logical_size_bytes"] == 800_000_000
    assert item["sample"]["sampled_elements"] <= 17
    assert item["sample"]["sample_selection"] != [[0, 10_000, None], [0, 10_000, None]]


@pytest.mark.parametrize("shape", [(0,), (0, 5), (3, 0, 4)])
def test_empty_dataset_statistics(tmp_path: Path, shape: tuple[int, ...]) -> None:
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("empty", shape=shape, dtype="f4")
    item = ai_data.inspect_hdf5(path)["datasets"][0]
    sample = item["sample"]
    assert sample["sampled_elements"] == 0
    assert sample["sample_selection"] == [[0, 0, None]] * len(shape)
    assert sample["all_finite"] is None
    assert sample["finite_count"] == 0
    assert sample["nan_count"] == 0
    assert sample["positive_infinity_count"] == 0
    assert sample["negative_infinity_count"] == 0
    assert sample["minimum"] is None
    assert sample["maximum"] is None
    assert sample["mean"] is None


def test_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ai_data.inspect_hdf5(tmp_path / "missing.h5")
    invalid = tmp_path / "invalid.h5"
    invalid.write_text("not HDF5", encoding="utf-8")
    with pytest.raises(ai_data.HDF5InspectionError, match="Invalid or unreadable"):
        ai_data.inspect_hdf5(invalid)


def test_read_only_no_mutation(sample_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before_digest = digest(sample_file)
    before_size = sample_file.stat().st_size
    original_file = h5py.File
    modes = []

    def recording_file(*args, **kwargs):
        modes.append(args[1] if len(args) > 1 else kwargs.get("mode"))
        return original_file(*args, **kwargs)

    monkeypatch.setattr(ai_data.h5py, "File", recording_file)
    ai_data.inspect_hdf5(sample_file)
    assert modes == ["r"]
    assert digest(sample_file) == before_digest
    assert sample_file.stat().st_size == before_size
    with original_file(sample_file, "r") as handle:
        assert np.array_equal(handle["measurements/temperature_C"][:], np.arange(24).reshape(2, 3, 4))


def test_ambiguous_semantics_are_not_selected() -> None:
    result = ai_data.infer_semantic_keys(["/a/temperature", "/b/temperature"])
    assert result["temperature"]["selected"] is None
    assert result["temperature"]["certainty"] == "ambiguous"
    assert len(result["temperature"]["alternatives"]) == 2


def test_semantic_short_tokens_use_basename_boundaries() -> None:
    heat_flux = ai_data.infer_semantic_keys(["/measurements/heat_flux"])
    assert heat_flux["heat_flux"]["selected"] == "/measurements/heat_flux"
    assert heat_flux["temperature"]["selected"] is None
    assert heat_flux["temperature"]["alternatives"] == []

    result = ai_data.infer_semantic_keys(["/T_DF", "/group/HF_DF", "/group/x", "/group/y"])
    assert result["temperature"]["selected"] == "/T_DF"
    assert result["heat_flux"]["selected"] == "/group/HF_DF"
    assert result["x_coordinate"]["selected"] == "/group/x"
    assert result["y_coordinate"]["selected"] == "/group/y"
    assert result["temperature"]["alternatives"][0]["rule"] == "basename prefix matches 't_'"
    assert result["heat_flux"]["alternatives"][0]["rule"] == "basename prefix matches 'hf_'"


def test_unrelated_names_are_not_semantic_short_token_matches() -> None:
    result = ai_data.infer_semantic_keys(
        ["/measurements/heat_transfer", "/group/growth_factor", "/group/xy_data"]
    )
    assert result["temperature"]["alternatives"] == []
    assert result["heat_flux"]["alternatives"] == []
    assert result["x_coordinate"]["alternatives"] == []
    assert result["y_coordinate"]["alternatives"] == []


def test_descriptive_semantic_names_remain_supported() -> None:
    paths = [
        "/temperature_C",
        "/heat_flux_kW_m2",
        "/time_s",
        "/x_coordinate_m",
        "/y_coordinate_m",
    ]
    result = ai_data.infer_semantic_keys(paths)
    assert result["temperature"]["selected"] == "/temperature_C"
    assert result["heat_flux"]["selected"] == "/heat_flux_kW_m2"
    assert result["time"]["selected"] == "/time_s"
    assert result["x_coordinate"]["selected"] == "/x_coordinate_m"
    assert result["y_coordinate"]["selected"] == "/y_coordinate_m"


def test_local_inspection_uses_no_network(
    sample_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_network(*args, **kwargs):
        raise AssertionError("unexpected real network access")

    monkeypatch.setattr(ai_data.urllib.request, "urlopen", reject_network)
    assert ai_data.inspect_hdf5(sample_file)["datasets"]


def test_sample_limit_validation(sample_file: Path) -> None:
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            ai_data.inspect_hdf5(sample_file, sample_limit=invalid)


class FakeResponse:
    def __init__(self, status: int, content_range: str, body: bytes):
        self.status = status
        self.headers = {"Content-Range": content_range}
        self._body = io.BytesIO(body)

    def getcode(self):
        return self.status

    def read(self, size=-1):
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return next(self.responses)


def test_content_range_parsing_and_validation() -> None:
    assert ai_data.parse_content_range("bytes 10-19/100") == (10, 19, 100)
    for invalid in ("bytes 20-10/100", "bytes 0-100/100", "items 0-1/2"):
        with pytest.raises(ai_data.RangeDownloadError):
            ai_data.parse_content_range(invalid)


@pytest.mark.parametrize(
    ("status", "content_range", "body"),
    [
        (200, "bytes 0-3/10", b"abcd"),
        (206, "bytes 1-4/10", b"abcd"),
        (206, "bytes 0-3/10", b"abc"),
        (206, "bytes 0-3/10", b"abcde"),
    ],
)
def test_range_download_rejects_invalid_response(
    tmp_path: Path, status: int, content_range: str, body: bytes
) -> None:
    opener = FakeOpener([FakeResponse(status, content_range, body)])
    with pytest.raises(ai_data.RangeDownloadError):
        ai_data.download_exact_range(
            ai_data.OFFICIAL_ULRI_URL, 0, 3, tmp_path / "member.part",
            member_name="member", max_response_bytes=4, retries=1, opener=opener,
        )
    assert not (tmp_path / "member.part").read_bytes()


def test_range_download_resume_offset(tmp_path: Path) -> None:
    part = tmp_path / "member.part"
    part.write_bytes(b"ab")
    opener = FakeOpener([FakeResponse(206, "bytes 12-13/100", b"cd")])
    result = ai_data.download_exact_range(
        ai_data.OFFICIAL_ULRI_URL, 10, 13, part,
        member_name="member", max_response_bytes=4, chunk_size=2, opener=opener,
    )
    assert part.read_bytes() == b"abcd"
    assert result["completed_bytes"] == 4
    assert result["resumed_bytes"] == 2
    assert opener.requests[0][0].headers["Range"] == "bytes=12-13"
    assert opener.requests[0][0].get_header("Accept-encoding") == "identity"
    part.write_bytes(b"12345")
    with pytest.raises(ai_data.RangeDownloadError, match="oversized"):
        ai_data.resume_offset(part, 4)


def test_range_download_rejects_outer_size_change(tmp_path: Path) -> None:
    opener = FakeOpener([FakeResponse(206, "bytes 0-3/99", b"abcd")])
    with pytest.raises(ai_data.RangeDownloadError, match="Outer-file size mismatch"):
        ai_data.download_exact_range(
            ai_data.OFFICIAL_ULRI_URL, 0, 3, tmp_path / "member.part",
            member_name="member", max_response_bytes=4, expected_total_size=100,
            retries=1, opener=opener,
        )


def test_range_download_budget_rejects_before_io_or_network(tmp_path: Path) -> None:
    destination = tmp_path / "not-created" / "member.part"
    opener = FakeOpener([])
    with pytest.raises(
        ai_data.RangeDownloadError,
        match="Requested range is 5 bytes, exceeding the permitted 4 response-body bytes",
    ):
        ai_data.download_exact_range(
            ai_data.OFFICIAL_ULRI_URL, 10, 14, destination,
            member_name="member", max_response_bytes=4, opener=opener,
        )
    assert opener.requests == []
    assert not destination.exists()
    assert not destination.parent.exists()


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_range_download_rejects_invalid_budget(tmp_path: Path, budget) -> None:
    opener = FakeOpener([])
    with pytest.raises(ValueError, match="max_response_bytes must be a positive integer"):
        ai_data.download_exact_range(
            ai_data.OFFICIAL_ULRI_URL, 0, 0, tmp_path / "member.part",
            member_name="member", max_response_bytes=budget, opener=opener,
        )
    assert opener.requests == []
    assert not (tmp_path / "member.part").exists()


def test_resumed_range_cannot_bypass_full_range_budget(tmp_path: Path) -> None:
    destination = tmp_path / "member.part"
    destination.write_bytes(b"abcd")
    opener = FakeOpener([])
    with pytest.raises(ai_data.RangeDownloadError, match="Requested range is 5 bytes"):
        ai_data.download_exact_range(
            ai_data.OFFICIAL_ULRI_URL, 10, 14, destination,
            member_name="member", max_response_bytes=1, opener=opener,
        )
    assert destination.read_bytes() == b"abcd"
    assert opener.requests == []


def _raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def test_verified_raw_deflate_atomic_extraction(tmp_path: Path) -> None:
    payload = ai_data.HDF5_SIGNATURE + b"verified payload"
    compressed = _raw_deflate(payload)
    source = tmp_path / "file.deflate.part"
    output = tmp_path / "file.h5"
    source.write_bytes(compressed)
    result = ai_data.extract_raw_deflate(
        source, output, expected_compressed_size=len(compressed),
        expected_uncompressed_size=len(payload), expected_crc32=zlib.crc32(payload),
        block_size=3,
    )
    assert output.read_bytes() == payload
    assert not (tmp_path / "file.h5.part").exists()
    assert result["crc32"] == f"{zlib.crc32(payload):08x}"


@pytest.mark.parametrize("failure", ["size", "crc", "signature"])
def test_extraction_failure_preserves_temporary_data(tmp_path: Path, failure: str) -> None:
    payload = ai_data.HDF5_SIGNATURE + b"payload"
    if failure == "signature":
        payload = b"not hdf5 payload"
    compressed = _raw_deflate(payload)
    source = tmp_path / "file.deflate.part"
    output = tmp_path / "file.h5"
    source.write_bytes(compressed)
    expected_size = len(payload) + (1 if failure == "size" else 0)
    expected_crc = zlib.crc32(payload) ^ (1 if failure == "crc" else 0)
    with pytest.raises(ai_data.DeflateVerificationError):
        ai_data.extract_raw_deflate(
            source, output, expected_compressed_size=len(compressed),
            expected_uncompressed_size=expected_size, expected_crc32=expected_crc,
        )
    assert source.exists()
    assert (tmp_path / "file.h5.part").exists()
    assert not output.exists()


def test_chunked_statistics_and_pair_alignment(tmp_path: Path) -> None:
    temperature = tmp_path / "temperature.h5"
    heat_flux = tmp_path / "heat_flux.h5"
    values = np.array([[[1.0, 2.0]], [[np.nan, np.inf]], [[3.0, -np.inf]]])
    before = {}
    for path, key in ((temperature, "temperature"), (heat_flux, "heat_flux")):
        with h5py.File(path, "w") as handle:
            handle.create_dataset(key, data=values)
            handle.create_dataset("time", data=[0.0, 1.0, 2.0])
            handle.create_dataset("x", data=[0.0, 1.0])
            handle.create_dataset("y", data=[0.0])
        before[path] = digest(path)
    with h5py.File(temperature, "r") as handle:
        stats = ai_data.chunked_dataset_statistics(handle["temperature"], frame_chunk=1)
    assert stats["finite_count"] == 3
    assert stats["non_finite_count"] == 3
    assert stats["nan_count"] == 1
    assert stats["positive_infinity_count"] == 1
    assert stats["negative_infinity_count"] == 1
    assert stats["mean"] == 2.0
    assert stats["standard_deviation"] == pytest.approx((2 / 3) ** 0.5)
    alignment = ai_data.assess_pair_alignment(
        temperature, heat_flux, temperature_key="temperature", heat_flux_key="heat_flux",
        time_key="time", x_key="x", y_key="y",
    )
    assert alignment["shape_equal"] is True
    assert alignment["time"]["values_equal"] is True
    assert all(digest(path) == checksum for path, checksum in before.items())


@pytest.mark.parametrize("family", ai_data.SOURCE_FAMILIES)
def test_every_source_family_is_deterministic_and_shaped(family: str) -> None:
    grid = ai_data.Grid2D(7, 8)
    first, metadata = ai_data.construct_source(grid, family, seed=123)
    second, _ = ai_data.construct_source(grid, family, seed=123)
    different, _ = ai_data.construct_source(grid, family, seed=124)
    assert first.shape == grid.shape
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert metadata["family"] == family
    assert np.min(first) >= 0.0


def test_signed_source_requires_opt_in() -> None:
    grid = ai_data.Grid2D(6, 6)
    with pytest.raises(ValueError, match="allow_signed"):
        ai_data.construct_source(grid, "one_gaussian", seed=1, signed=True)
    signed, metadata = ai_data.construct_source(
        grid, "one_gaussian", seed=1, signed=True, allow_signed=True
    )
    assert np.min(signed) < 0 < np.max(signed)
    assert metadata["signed"] is True


def test_sample_generation_schema_masks_baselines_and_stable_id(pipeline_config: dict) -> None:
    plan = ai_data.plan_synthetic_samples(pipeline_config)[0]
    first = ai_data.generate_synthetic_sample(pipeline_config, plan)
    second = ai_data.generate_synthetic_sample(pipeline_config, plan)
    assert first["sample_id"] == second["sample_id"]
    assert np.array_equal(first["true_source"], second["true_source"])
    for field in ("true_source", "full_temperature", "identity_reconstruction", "smoothness_reconstruction"):
        assert first[field].shape == (6, 6)
    indices = first["sensor_indices"]
    assert first["sensor_mask"].sum() == first["sensor_count"]
    assert first["sensor_mask"][indices[:, 0], indices[:, 1]].all()
    assert np.array_equal(first["sparse_temperature"][indices[:, 0], indices[:, 1]], first["measured_temperatures"])
    changed = ai_data.generate_synthetic_sample(pipeline_config, dict(plan, source_seed=plan["source_seed"] + 1))
    assert changed["sample_id"] != first["sample_id"]
    assert not np.array_equal(changed["true_source"], first["true_source"])


def test_split_plan_is_reproducible_and_leakage_safe(pipeline_config: dict) -> None:
    first = ai_data.plan_synthetic_samples(pipeline_config)
    assert first == ai_data.plan_synthetic_samples(pipeline_config)
    assert {item["split"] for item in first} == set(ai_data.SPLIT_ROLES)
    assert len({item["source_seed"] for item in first}) == len(first)
    training = [item for item in first if item["split"] == "train"]
    assert all(item["source_family"] != "sharp_edged" for item in training)
    assert all(item["sensor_strategy"] != "center_focused" for item in training)
    assert all(item["noise_level"] != 0.1 for item in training)
    assert next(i for i in first if i["split"] == "test_ood_shape")["source_family"] == "sharp_edged"
    assert next(i for i in first if i["split"] == "test_ood_sensor")["sensor_strategy"] == "center_focused"
    assert next(i for i in first if i["split"] == "test_ood_noise")["noise_level"] == 0.1


def test_training_only_normalization_extremes_zero_variance_and_inverse() -> None:
    samples = [
        {"sample_id": "train", "split": "train", "value": np.array([1.0, 3.0]), "constant": np.ones(2)},
        {"sample_id": "validation", "split": "validation", "value": np.array([1e100]), "constant": np.array([9e99])},
        {"sample_id": "test", "split": "test_id", "value": np.array([-1e100]), "constant": np.array([-9e99])},
    ]
    report = ai_data.fit_training_normalization(samples, fields=("value", "constant"))
    assert report["sample_ids"] == ["train"]
    assert report["statistics"]["value"]["mean"] == 2.0
    constant = report["statistics"]["constant"]
    assert constant["zero_variance"] is True and constant["scale"] == 1.0
    original = np.array([1.0, 2.0, 3.0])
    restored = ai_data.inverse_normalize_array(ai_data.normalize_array(original, report["statistics"]["value"]), report["statistics"]["value"])
    assert np.allclose(restored, original)
    with pytest.raises(ai_data.DatasetPipelineError, match="without training"):
        ai_data.fit_training_normalization(samples[1:], fields=("value",))


def test_tiny_dataset_atomic_storage_lazy_read_and_no_input_mutation(tmp_path: Path, pipeline_config: dict) -> None:
    output = tmp_path / "dataset"
    report = ai_data.build_synthetic_dataset(pipeline_config, output_directory=output)
    assert report["sample_count"] == 6
    assert not (output / "synthetic_dataset.h5.part").exists()
    assert all(value == 1 for value in report["split_counts"].values())
    with h5py.File(output / "synthetic_dataset.h5", "r") as handle:
        assert handle.mode == "r" and handle["true_source"].compression == "gzip"
        assert "observation_matrix" not in handle
    before = digest(output / "synthetic_dataset.h5")
    with ai_data.SyntheticDatasetReader(output) as reader:
        first = reader[0]
        stored = first["true_source"].copy()
        first["true_source"][...] = -999
        assert np.array_equal(reader[0]["true_source"], stored)
    assert digest(output / "synthetic_dataset.h5") == before
    normalization = json.loads((output / "normalization.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert normalization["sample_ids"] == [item["sample_id"] for item in manifest["samples"] if item["split"] == "train"]
    assert ai_data.validate_synthetic_dataset(output)["valid"] is True


def test_partial_and_corrupt_synthetic_outputs_are_rejected(tmp_path: Path, pipeline_config: dict) -> None:
    output = tmp_path / "partial"
    output.mkdir()
    (output / "synthetic_dataset.h5.part").write_bytes(b"partial")
    with pytest.raises(ai_data.DatasetPipelineError, match="exists or is partial"):
        ai_data.build_synthetic_dataset(pipeline_config, output_directory=output)
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "manifest.json").write_text("{}")
    with pytest.raises(ai_data.DatasetPipelineError, match="artifact is missing"):
        ai_data.SyntheticDatasetReader(corrupt)


def test_external_manifest_labels_patterns_shapes_and_read_only(tmp_path: Path) -> None:
    temperature, flux = make_external_pair(tmp_path)
    before = (digest(temperature), digest(flux))
    manifest_path = tmp_path / "external.json"
    manifest = ai_data.build_external_manifest(temperature, flux, manifest_path, experiment_id="E-TM-F", plate_id="PR", calculate_checksums=False)
    assert manifest["frame_indices"] == [0, 1, 2, 3, 4] and manifest["frame_shape"] == [6, 7]
    assert manifest["task_type"] == "external_heat_flux" and manifest["usage_role"] == "external_audit"
    assert manifest["target_provenance"].startswith("HFITS-derived")
    assert manifest["independent_ground_truth"] is False and manifest["classical_q_target"] is False
    assert manifest["time_values"] is None and manifest["temperature_units"] == "unresolved"
    assert manifest["x_coordinates"] is None
    assert ai_data.validate_external_manifest(manifest_path)["valid"] is True
    assert (digest(temperature), digest(flux)) == before


def test_external_manifest_rejects_missing_indices_and_shape_mismatch(tmp_path: Path) -> None:
    temperature, flux = make_external_pair(tmp_path, frames=(0, 2))
    with pytest.raises(ai_data.DatasetPipelineError, match="Missing temperature"):
        ai_data.build_external_manifest(temperature, flux, tmp_path / "bad.json", experiment_id="E", plate_id="P", calculate_checksums=False)
    other = tmp_path / "other"
    other.mkdir()
    temperature2, flux2 = make_external_pair(other, frames=(0, 1))
    with h5py.File(flux2, "a") as handle:
        del handle["estimated_flux_batch0_frame000001"]
        handle.create_dataset("estimated_flux_batch0_frame000001", data=np.ones((3, 3)))
    with pytest.raises(ai_data.DatasetPipelineError, match="shapes differ"):
        ai_data.build_external_manifest(temperature2, flux2, tmp_path / "bad2.json", experiment_id="E", plate_id="P", calculate_checksums=False)


@pytest.mark.parametrize(("policy", "expected"), [("edge-repeat", [0, 0, 1]), ("reflect", [1, 0, 1])])
def test_external_window_boundaries_sensor_determinism_and_no_mutation(tmp_path: Path, policy: str, expected: list[int]) -> None:
    temperature, flux = make_external_pair(tmp_path)
    manifest_path = tmp_path / "external.json"
    ai_data.build_external_manifest(temperature, flux, manifest_path, experiment_id="E-TM-F", plate_id="PR", calculate_checksums=False)
    reader = ai_data.ExternalWindowReader(manifest_path, window_size=3, boundary_policy=policy, sensor_strategy="random", sensor_count=5, sensor_seed=22)
    before = (digest(temperature), digest(flux))
    with reader:
        first, second = reader.get(0), reader.get(0)
    assert first["window_frame_indices"] == expected and first["temperature_window"].shape == (3, 6, 7)
    assert first["external_reference_flux"].shape == (6, 7)
    assert np.array_equal(first["sensor_indices"], second["sensor_indices"])
    assert first["sensor_mask"].sum() == 5 and first["time_values"] is None
    assert (digest(temperature), digest(flux)) == before


def test_external_window_validation_and_missing_frame(tmp_path: Path) -> None:
    temperature, flux = make_external_pair(tmp_path)
    manifest_path = tmp_path / "external.json"
    ai_data.build_external_manifest(temperature, flux, manifest_path, experiment_id="E-TM-F", plate_id="PR", calculate_checksums=False)
    with pytest.raises(ValueError, match="odd"):
        ai_data.ExternalWindowReader(manifest_path, window_size=2, boundary_policy="reject", sensor_strategy="random", sensor_count=4, sensor_seed=1)
    reader = ai_data.ExternalWindowReader(manifest_path, window_size=3, boundary_policy="reject", sensor_strategy="random", sensor_count=4, sensor_seed=1)
    with pytest.raises(ai_data.DatasetPipelineError, match="boundary"):
        reader.get(0)
    with pytest.raises(ai_data.DatasetPipelineError, match="missing"):
        reader.get(99)

    boundary_reader = ai_data.ExternalWindowReader(
        manifest_path, window_size=1, boundary_policy="reject",
        sensor_strategy="boundary", sensor_count=4, sensor_seed=1,
    )
    with boundary_reader:
        boundary_sample = boundary_reader.get(2)
    assert boundary_sample["sensor_mask"].sum() == 4


def test_experiment_level_split_keeps_all_frames_and_plates_together() -> None:
    records = [{"experiment_id": experiment, "plate_id": plate, "frame": frame} for experiment in ("E1", "E2", "E3", "E4") for plate in ("A", "B") for frame in range(3)]
    first = ai_data.assign_experiment_splits(records, seed=7)
    assert first == ai_data.assign_experiment_splits(list(reversed(records)), seed=7)
    assert set(first) == {"E1", "E2", "E3", "E4"} and set(first.values()) == {"train", "validation", "test"}
    with pytest.raises(ai_data.DatasetPipelineError, match="1 experiment"):
        ai_data.assign_experiment_splits([{"experiment_id": "E-TM-F"}], seed=7)
    assert ai_data.assign_experiment_splits([{"experiment_id": "E-TM-F"}], seed=7, split_names=("external_audit",)) == {"E-TM-F": "external_audit"}


def test_configuration_validation_and_default_preflight(pipeline_config: dict) -> None:
    assert ai_data.validate_dataset_config(pipeline_config) is pipeline_config
    for mutation, message in (
        ({"sensor_strategies": ["regular_grid", "boundary"]}, "boundary"),
        ({"sensor_strategies": ["random", "random"]}, "unique"),
        ({"ood_sensor_strategies": ["center_focused", "boundary"]}, "available"),
        ({"sensor_count_range": [1, 17]}, "interior nodes"),
    ):
        invalid = dict(pipeline_config)
        invalid.update(mutation)
        with pytest.raises(ai_data.DatasetPipelineError, match=message):
            ai_data.validate_dataset_config(invalid)
    default_path = Path(__file__).parents[1] / "configs" / "dataset_default.json"
    default = ai_data.load_dataset_config(default_path)
    plans = ai_data.plan_synthetic_samples(default)
    capacity = (default["grid_shape"][0] - 2) * (default["grid_shape"][1] - 2)
    assert len(plans) == 1200
    assert all(plan["sensor_strategy"] in ai_data.SYNTHETIC_SENSOR_STRATEGIES for plan in plans)
    assert all(1 <= plan["sensor_count"] <= capacity for plan in plans)
    assert {p["sensor_strategy"] for p in plans if p["split"] == "train"} <= {"regular_grid", "random"}


def test_effective_source_boundaries_mask_forward_and_input_immutability(pipeline_config: dict) -> None:
    plan = ai_data.plan_synthetic_samples(pipeline_config)[0]
    original, _ = ai_data.construct_source(
        ai_data.Grid2D(6, 6), plan["source_family"], seed=plan["source_seed"],
        amplitude_range=tuple(pipeline_config["amplitude_range"]),
        width_range=tuple(pipeline_config["width_range"]),
        size_range=tuple(pipeline_config["size_range"]),
        source_count_range=tuple(pipeline_config["source_count_range"]),
    )
    before = original.copy()
    sample = ai_data.generate_synthetic_sample(pipeline_config, plan)
    assert np.array_equal(original, before)
    source = sample["true_source"]
    assert np.all(source[[0, -1], :] == 0) and np.all(source[:, [0, -1]] == 0)
    assert np.all(sample["identity_reconstruction"][[0, -1], :] == 0)
    assert np.all(sample["smoothness_reconstruction"][:, [0, -1]] == 0)
    assert np.array_equal(ai_data.solve_forward(source, ai_data.Grid2D(6, 6)), sample["full_temperature"])
    mask = ai_data.source_valid_mask((6, 6))
    assert mask[1:-1, 1:-1].all() and not mask[[0, -1], :].any() and not mask[:, [0, -1]].any()
    assert sample["source_target_domain"] == "interior_nodes"


def test_multiple_gaussian_metadata_exactly_reconstructs_field() -> None:
    grid = ai_data.Grid2D(9, 10)
    field, metadata = ai_data.construct_source(grid, "multiple_gaussians", seed=44)
    rebuilt = np.zeros(grid.shape)
    for hotspot in metadata["hotspots"]:
        rebuilt += ai_data.gaussian_source(
            grid, center=tuple(hotspot["center"]),
            amplitude=hotspot["amplitude"], sigma=hotspot["sigma"],
        )
    assert np.array_equal(field, rebuilt)
    assert metadata["count"] == len(metadata["hotspots"])
    assert metadata["generator"] == "sum_of_thermoreconlab_gaussian_source"


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("source_seed", 999), ("sensor_count", 5), ("identity_alpha", 0.02),
        ("smoothness_alpha", 0.002), ("noise_level", 0.02),
    ],
)
def test_sample_id_changes_with_physical_metadata(
    pipeline_config: dict, change: str, value: object
) -> None:
    plan = ai_data.plan_synthetic_samples(pipeline_config)[0]
    baseline = ai_data.generate_synthetic_sample(pipeline_config, plan)
    changed_plan = dict(plan)
    changed_plan[change] = value
    changed = ai_data.generate_synthetic_sample(pipeline_config, changed_plan)
    assert changed["sample_id"] != baseline["sample_id"]
    moved = ai_data.generate_synthetic_sample(
        pipeline_config, dict(plan, split="test_id", split_reason="manifest move")
    )
    assert moved["sample_id"] == baseline["sample_id"]


def test_smoke_plan_sample_ids_are_unique(pipeline_config: dict) -> None:
    samples = [
        ai_data.generate_synthetic_sample(pipeline_config, plan)
        for plan in ai_data.plan_synthetic_samples(pipeline_config)
    ]
    assert len({sample["sample_id"] for sample in samples}) == len(samples)


def test_atomic_sidecar_integrity_and_existing_output_rejection(tmp_path: Path, pipeline_config: dict) -> None:
    output = tmp_path / "dataset"
    report = ai_data.build_synthetic_dataset(pipeline_config, output_directory=output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dataset_sha256"] == digest(output / "synthetic_dataset.h5")
    assert report["manifest_content_sha256"] == manifest["manifest_content_sha256"]
    with pytest.raises(ai_data.DatasetPipelineError, match="already exists"):
        ai_data.build_synthetic_dataset(pipeline_config, output_directory=output)


@pytest.mark.parametrize(
    ("missing", "message"),
    [("normalization.json", "normalization"), ("configuration.json", "configuration")],
)
def test_reader_rejects_missing_sidecars(
    tmp_path: Path, pipeline_config: dict, missing: str, message: str
) -> None:
    output = tmp_path / missing.replace(".json", "")
    ai_data.build_synthetic_dataset(pipeline_config, output_directory=output)
    (output / missing).unlink()
    with pytest.raises(ai_data.DatasetPipelineError, match=message):
        ai_data.SyntheticDatasetReader(output)


@pytest.mark.parametrize("artifact", ["configuration.json", "normalization.json"])
def test_reader_rejects_tampered_sidecars(tmp_path: Path, pipeline_config: dict, artifact: str) -> None:
    output = tmp_path / artifact.replace(".json", "")
    ai_data.build_synthetic_dataset(pipeline_config, output_directory=output)
    payload = json.loads((output / artifact).read_text())
    payload["tampered"] = True
    (output / artifact).write_text(json.dumps(payload))
    with pytest.raises(ai_data.DatasetPipelineError, match="checksum"):
        ai_data.SyntheticDatasetReader(output)


def test_reader_rejects_tampered_hdf5(tmp_path: Path, pipeline_config: dict) -> None:
    output = tmp_path / "hdf5"
    ai_data.build_synthetic_dataset(pipeline_config, output_directory=output)
    with h5py.File(output / "synthetic_dataset.h5", "a") as handle:
        handle["true_source"][0, 2, 2] += 1
    with pytest.raises(ai_data.DatasetPipelineError, match="HDF5 checksum"):
        ai_data.SyntheticDatasetReader(output)


def test_failure_before_manifest_publication_leaves_no_manifest(
    tmp_path: Path, pipeline_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failure"
    original_replace = Path.replace

    def failing_replace(self: Path, target: Path) -> Path:
        if self.name == "normalization.json.part":
            raise OSError("simulated publication failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated"):
        ai_data.build_synthetic_dataset(pipeline_config, output_directory=output)
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("corruption", ["count", "mask", "index", "sparse"])
def test_reader_rejects_internally_inconsistent_samples(
    tmp_path: Path, pipeline_config: dict, corruption: str
) -> None:
    base = tmp_path / "base"
    ai_data.build_synthetic_dataset(pipeline_config, output_directory=base)
    output = tmp_path / corruption
    shutil.copytree(base, output)
    with h5py.File(output / "synthetic_dataset.h5", "a") as handle:
        if corruption == "count":
            handle["sensor_count"][0] += 1
        elif corruption == "mask":
            handle["sensor_mask"][0, 0, 0] = not handle["sensor_mask"][0, 0, 0]
        elif corruption == "index":
            handle["sensor_indices"][0, 0] = [99, 99]
        else:
            indices = handle["sensor_indices"][0, 0]
            handle["sparse_temperature"][0, indices[0], indices[1]] += 1
    resign_manifest(output)
    with pytest.raises(ai_data.DatasetPipelineError, match=corruption if corruption != "count" else "sensor_count"):
        with ai_data.SyntheticDatasetReader(output) as reader:
            reader[0]


@pytest.mark.parametrize("policy", ["edge-repeat", "reflect"])
def test_one_frame_window_repeats_without_loop(policy: str) -> None:
    assert ai_data._window_indices(0, 1, 5, policy) == [0, 0, 0, 0, 0]
    with pytest.raises(ai_data.DatasetPipelineError, match="boundary"):
        ai_data._window_indices(0, 1, 3, "reject")
    with pytest.raises(ValueError, match="count"):
        ai_data._window_indices(0, 0, 1, "reflect")
    with pytest.raises(ValueError, match="center"):
        ai_data._window_indices(1, 1, 1, "reflect")


def test_external_full_validation_rejects_file_and_manifest_tampering(tmp_path: Path) -> None:
    temperature, flux = make_external_pair(tmp_path)
    manifest_path = tmp_path / "external.json"
    ai_data.build_external_manifest(
        temperature, flux, manifest_path, experiment_id="E", plate_id="P",
        calculate_checksums=True,
    )
    assert ai_data.validate_external_manifest(manifest_path)["valid"] is True
    with h5py.File(temperature, "a") as handle:
        handle["surface_temperature_batch0_frame000000"][0, 0] += 1
    with pytest.raises(ai_data.DatasetPipelineError, match="temperature_path"):
        ai_data.validate_external_manifest(manifest_path)


@pytest.mark.parametrize(("field", "value"), [("task_type", "q"), ("usage_role", "train")])
def test_external_validation_rejects_semantic_manifest_changes(
    tmp_path: Path, field: str, value: str
) -> None:
    temperature, flux = make_external_pair(tmp_path)
    manifest_path = tmp_path / "external.json"
    ai_data.build_external_manifest(temperature, flux, manifest_path, experiment_id="E", plate_id="P", calculate_checksums=False)
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    resign_external_manifest(manifest_path)
    with pytest.raises(ai_data.DatasetPipelineError, match=field):
        ai_data.validate_external_manifest(manifest_path)


def test_external_validation_rejects_wrong_size_flux_change_and_changed_keys(tmp_path: Path) -> None:
    temperature, flux = make_external_pair(tmp_path)
    manifest_path = tmp_path / "external.json"
    ai_data.build_external_manifest(temperature, flux, manifest_path, experiment_id="E", plate_id="P", calculate_checksums=False)
    manifest = json.loads(manifest_path.read_text())
    manifest["temperature_size_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest))
    resign_external_manifest(manifest_path)
    with pytest.raises(ai_data.DatasetPipelineError, match="size mismatch"):
        ai_data.validate_external_manifest(manifest_path)

    temperature2, flux2 = make_external_pair(tmp_path / "changed")
    second_manifest = tmp_path / "changed.json"
    ai_data.build_external_manifest(temperature2, flux2, second_manifest, experiment_id="E", plate_id="P", calculate_checksums=True)
    with h5py.File(flux2, "a") as handle:
        handle["estimated_flux_batch0_frame000000"][0, 0] += 1
    with pytest.raises(ai_data.DatasetPipelineError, match="heat_flux_path"):
        ai_data.validate_external_manifest(second_manifest)

    temperature3, flux3 = make_external_pair(tmp_path / "keys")
    third_manifest = tmp_path / "keys.json"
    ai_data.build_external_manifest(temperature3, flux3, third_manifest, experiment_id="E", plate_id="P", calculate_checksums=False)
    with h5py.File(temperature3, "a") as handle:
        handle.move("surface_temperature_batch0_frame000004", "changed_frame")
    third = json.loads(third_manifest.read_text())
    third["temperature_size_bytes"] = temperature3.stat().st_size
    third_manifest.write_text(json.dumps(third))
    resign_external_manifest(third_manifest)
    with pytest.raises(ai_data.DatasetPipelineError, match="Unexpected temperature HDF5 keys"):
        ai_data.validate_external_manifest(third_manifest)


def test_external_validation_rejects_changed_frame_shape(tmp_path: Path) -> None:
    temperature, flux = make_external_pair(tmp_path)
    manifest_path = tmp_path / "external.json"
    ai_data.build_external_manifest(
        temperature, flux, manifest_path, experiment_id="E", plate_id="P",
        calculate_checksums=False,
    )
    with h5py.File(flux, "a") as handle:
        del handle["estimated_flux_batch0_frame000003"]
        handle.create_dataset("estimated_flux_batch0_frame000003", data=np.ones((3, 3)))
    manifest = json.loads(manifest_path.read_text())
    manifest["heat_flux_size_bytes"] = flux.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    resign_external_manifest(manifest_path)
    with pytest.raises(ai_data.DatasetPipelineError, match="shapes differ"):
        ai_data.validate_external_manifest(manifest_path)
