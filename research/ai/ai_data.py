"""Read-only, bounded HDF5 inspection for optional Phase 5 research."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from thermoreconlab.core.grid import Grid2D
from thermoreconlab.data import gaussian_source
from thermoreconlab.reconstruction import (
    reconstruct_smooth_tikhonov,
    reconstruct_tikhonov,
    solve_forward,
)
from thermoreconlab.sensors import (
    SensorData,
    add_gaussian_noise,
    boundary_sensors,
    center_focused_sensors,
    random_sensors,
    regular_grid_sensors,
    sample_field,
)


OFFICIAL_ULRI_URL = "https://ndownloader.figshare.com/files/59026160"
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
DATASET_SCHEMA_VERSION = 2
GENERATOR_VERSION = "phase5-task2-v2"
SYNTHETIC_SENSOR_STRATEGIES = ("regular_grid", "random", "center_focused")
EXTERNAL_SENSOR_STRATEGIES = (*SYNTHETIC_SENSOR_STRATEGIES, "boundary")
EXTERNAL_TEMPERATURE_PATTERN = re.compile(
    r"surface_temperature_batch0_frame(\d{6})"
)
EXTERNAL_FLUX_PATTERN = re.compile(r"estimated_flux_batch0_frame(\d{6})")


class HDF5InspectionError(ValueError):
    """Raised when a file cannot be inspected as HDF5."""


class RangeDownloadError(RuntimeError):
    """Raised when an exact HTTP byte-range download cannot be validated."""


class DeflateVerificationError(RuntimeError):
    """Raised when a raw-DEFLATE member fails integrity verification."""


class DatasetPipelineError(RuntimeError):
    """Raised when a generated dataset or external adapter is invalid."""


def parse_content_range(value: str) -> tuple[int, int, int]:
    """Parse an exact ``bytes start-end/total`` Content-Range value."""
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if match is None:
        raise RangeDownloadError(f"Invalid Content-Range: {value!r}")
    start, end, total = (int(part) for part in match.groups())
    if start > end or end >= total:
        raise RangeDownloadError(f"Invalid Content-Range bounds: {value!r}")
    return start, end, total


def resume_offset(part_path: str | Path, expected_length: int) -> int:
    """Return a validated number of already downloaded bytes."""
    path = Path(part_path)
    completed = path.stat().st_size if path.exists() else 0
    if completed > expected_length:
        raise RangeDownloadError(
            f"Partial file is oversized: {completed} > {expected_length} bytes"
        )
    return completed


def download_exact_range(
    url: str,
    start: int,
    end: int,
    part_path: str | Path,
    *,
    member_name: str,
    max_response_bytes: int,
    chunk_size: int = 64 * 1024 * 1024,
    timeout: float = 60.0,
    retries: int = 3,
    expected_total_size: int | None = None,
    opener: Any = None,
) -> dict[str, Any]:
    """Download one exact inclusive range using fresh redirect resolution per chunk."""
    if url != OFFICIAL_ULRI_URL:
        raise ValueError("Only the official ULRI Figshare URL is permitted")
    if start < 0 or end < start:
        raise ValueError("start and end must define a non-negative inclusive range")
    if (
        not isinstance(max_response_bytes, int)
        or isinstance(max_response_bytes, bool)
        or max_response_bytes < 1
    ):
        raise ValueError("max_response_bytes must be a positive integer")
    if chunk_size < 1 or retries < 1 or timeout <= 0:
        raise ValueError("chunk_size, retries, and timeout must be positive")
    expected_length = end - start + 1
    if expected_length > max_response_bytes:
        raise RangeDownloadError(
            f"Requested range is {expected_length} bytes, exceeding the permitted "
            f"{max_response_bytes} response-body bytes"
        )
    destination = Path(part_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = resume_offset(destination, expected_length)
    resumed_bytes = completed
    retry_events = 0
    started = time.monotonic()
    open_request = opener.open if opener is not None else urllib.request.urlopen
    with destination.open("ab") as output:
        while completed < expected_length:
            chunk_start = start + completed
            chunk_end = min(end, chunk_start + chunk_size - 1)
            expected_chunk = chunk_end - chunk_start + 1
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={chunk_start}-{chunk_end}",
                    "Accept-Encoding": "identity",
                },
                method="GET",
            )
            for attempt in range(1, retries + 1):
                try:
                    with open_request(request, timeout=timeout) as response:
                        status = getattr(response, "status", response.getcode())
                        if status != 206:
                            raise RangeDownloadError(
                                f"Expected HTTP 206 for {member_name}, received {status}"
                            )
                        header = response.headers.get("Content-Range", "")
                        actual_start, actual_end, total_size = parse_content_range(header)
                        if (actual_start, actual_end) != (chunk_start, chunk_end):
                            raise RangeDownloadError(
                                f"Content-Range mismatch for {member_name}: {header!r}"
                            )
                        if expected_total_size is not None and total_size != expected_total_size:
                            raise RangeDownloadError(
                                f"Outer-file size mismatch for {member_name}: "
                                f"{total_size} != {expected_total_size}"
                            )
                        body = response.read(expected_chunk + 1)
                        if len(body) != expected_chunk:
                            raise RangeDownloadError(
                                f"Chunk length mismatch for {member_name}: "
                                f"expected {expected_chunk}, received {len(body)}"
                            )
                    output.write(body)
                    output.flush()
                    completed += len(body)
                    percentage = 100.0 * completed / expected_length
                    print(
                        f"{member_name}: {completed}/{expected_length} bytes "
                        f"({percentage:.2f}%), chunk {chunk_start}-{chunk_end}"
                    )
                    break
                except (OSError, urllib.error.URLError) as exc:
                    if attempt == retries:
                        raise RangeDownloadError(
                            f"Range request failed after {retries} attempts: {exc}"
                        ) from exc
                    retry_events += 1
            else:  # pragma: no cover - loop always breaks or raises
                raise AssertionError("unreachable retry state")
    return {
        "member_name": member_name,
        "completed_bytes": completed,
        "expected_bytes": expected_length,
        "resumed_bytes": resumed_bytes,
        "retry_events": retry_events,
        "duration_seconds": time.monotonic() - started,
        "path": str(destination.resolve()),
    }


def extract_raw_deflate(
    compressed_path: str | Path,
    output_path: str | Path,
    *,
    expected_compressed_size: int,
    expected_uncompressed_size: int,
    expected_crc32: int,
    block_size: int = 1024 * 1024,
) -> dict[str, Any]:
    """Stream-decompress and atomically publish one verified raw-DEFLATE member."""
    source = Path(compressed_path)
    final_path = Path(output_path)
    temporary = final_path.with_name(final_path.name + ".part")
    actual_compressed = source.stat().st_size
    if actual_compressed != expected_compressed_size:
        raise DeflateVerificationError(
            f"Compressed size mismatch: {actual_compressed} != {expected_compressed_size}"
        )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    crc32 = 0
    written = 0
    signature = bytearray()
    try:
        with source.open("rb") as compressed, temporary.open("wb") as output:
            while True:
                block = compressed.read(block_size)
                if not block:
                    break
                decoded = decompressor.decompress(block)
                if decoded:
                    output.write(decoded)
                    written += len(decoded)
                    crc32 = zlib.crc32(decoded, crc32)
                    if len(signature) < len(HDF5_SIGNATURE):
                        signature.extend(decoded[: len(HDF5_SIGNATURE) - len(signature)])
            decoded = decompressor.flush()
            output.write(decoded)
            written += len(decoded)
            crc32 = zlib.crc32(decoded, crc32)
            if len(signature) < len(HDF5_SIGNATURE):
                signature.extend(decoded[: len(HDF5_SIGNATURE) - len(signature)])
        crc32 &= 0xFFFFFFFF
        if not decompressor.eof:
            raise DeflateVerificationError("Raw DEFLATE stream ended before end-of-stream marker")
        if written != expected_uncompressed_size:
            raise DeflateVerificationError(
                f"Uncompressed size mismatch: {written} != {expected_uncompressed_size}"
            )
        if crc32 != expected_crc32:
            raise DeflateVerificationError(
                f"CRC-32 mismatch: {crc32:08x} != {expected_crc32:08x}"
            )
        if bytes(signature) != HDF5_SIGNATURE:
            raise DeflateVerificationError(
                f"HDF5 signature mismatch: {bytes(signature).hex()}"
            )
        temporary.replace(final_path)
    except Exception:
        raise
    return {
        "compressed_bytes": actual_compressed,
        "uncompressed_bytes": written,
        "crc32": f"{crc32:08x}",
        "hdf5_signature": bytes(signature).hex(" "),
        "path": str(final_path.resolve()),
    }


def chunked_dataset_statistics(
    dataset: h5py.Dataset, *, frame_chunk: int = 16
) -> dict[str, Any]:
    """Calculate complete numeric statistics without loading a full dataset."""
    if frame_chunk < 1:
        raise ValueError("frame_chunk must be positive")
    if not np.issubdtype(dataset.dtype, np.number):
        raise TypeError("dataset must have a numeric dtype")
    selections = [()] if dataset.shape == () else [
        (slice(start, min(dataset.shape[0], start + frame_chunk)),)
        + (slice(None),) * (dataset.ndim - 1)
        for start in range(0, dataset.shape[0], frame_chunk)
    ]
    finite_count = nan_count = posinf_count = neginf_count = 0
    total = total_squares = 0.0
    minimum = maximum = None
    for selection in selections:
        values = np.asarray(dataset[selection], dtype=np.float64)
        finite = np.isfinite(values)
        finite_values = values[finite]
        finite_count += int(finite_values.size)
        nan_count += int(np.isnan(values).sum())
        posinf_count += int(np.isposinf(values).sum())
        neginf_count += int(np.isneginf(values).sum())
        if finite_values.size:
            current_min = float(finite_values.min())
            current_max = float(finite_values.max())
            minimum = current_min if minimum is None else min(minimum, current_min)
            maximum = current_max if maximum is None else max(maximum, current_max)
            total += float(finite_values.sum(dtype=np.float64))
            total_squares += float(np.square(finite_values).sum(dtype=np.float64))
    mean = total / finite_count if finite_count else None
    variance = max(0.0, total_squares / finite_count - mean * mean) if finite_count else None
    return {
        "finite_count": finite_count,
        "non_finite_count": nan_count + posinf_count + neginf_count,
        "nan_count": nan_count,
        "positive_infinity_count": posinf_count,
        "negative_infinity_count": neginf_count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "standard_deviation": variance ** 0.5 if variance is not None else None,
        "number_of_frames": int(dataset.shape[0]) if dataset.ndim else 1,
        "spatial_shape": list(dataset.shape[1:]) if dataset.ndim else [],
    }


def assess_pair_alignment(
    temperature_path: str | Path,
    heat_flux_path: str | Path,
    *,
    temperature_key: str,
    heat_flux_key: str,
    time_key: str | None = None,
    x_key: str | None = None,
    y_key: str | None = None,
) -> dict[str, Any]:
    """Compare pair shapes and optional coordinate arrays read-only."""
    with h5py.File(temperature_path, "r") as temperature_file, h5py.File(
        heat_flux_path, "r"
    ) as heat_flux_file:
        temperature_shape = tuple(temperature_file[temperature_key].shape)
        heat_flux_shape = tuple(heat_flux_file[heat_flux_key].shape)
        result = {
            "temperature_shape": list(temperature_shape),
            "heat_flux_shape": list(heat_flux_shape),
            "shape_equal": temperature_shape == heat_flux_shape,
        }
        for label, key in (("time", time_key), ("x", x_key), ("y", y_key)):
            if key is None or key not in temperature_file or key not in heat_flux_file:
                result[label] = "Unresolved or not present in both HDF5 files"
                continue
            first = np.asarray(temperature_file[key][...])
            second = np.asarray(heat_flux_file[key][...])
            result[label] = {
                "shape_equal": first.shape == second.shape,
                "values_equal": bool(np.array_equal(first, second, equal_nan=True)),
            }
    return result


def _json_value(value: Any) -> Any:
    """Convert HDF5 attribute values to stable JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _attributes(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in obj.attrs.items()}


def _sample_selection(shape: tuple[int, ...], limit: int) -> tuple[Any, ...]:
    """Return one bounded hyperslab selection containing at most ``limit`` values."""
    if not shape:
        return ()
    if 0 in shape:
        return tuple(slice(0, 0) for _ in shape)
    lengths = [1] * len(shape)
    remaining = limit
    for axis in range(len(shape) - 1, -1, -1):
        take = min(shape[axis], max(1, remaining))
        lengths[axis] = take
        remaining = max(1, remaining // take)
    return tuple(slice(0, length) for length in lengths)


def _sample_statistics(dataset: h5py.Dataset, limit: int) -> dict[str, Any]:
    selection = _sample_selection(dataset.shape, limit)
    if dataset.shape == ():
        sample = np.asarray(dataset[()])
    elif dataset.size == 0:
        sample = np.asarray([], dtype=dataset.dtype)
    else:
        sample = np.asarray(dataset[selection])
    flat = sample.reshape(-1)
    result: dict[str, Any] = {
        "sampled_elements": int(flat.size),
        "sample_limit": limit,
        "sample_selection": "scalar" if dataset.shape == () else [
            [part.start, part.stop, part.step]
            for part in selection
        ],
        "statistics_available": bool(np.issubdtype(dataset.dtype, np.number)),
    }
    if not result["statistics_available"]:
        return result
    values = flat.astype(np.float64, copy=False)
    nan = np.isnan(values)
    posinf = np.isposinf(values)
    neginf = np.isneginf(values)
    finite = np.isfinite(values)
    finite_values = values[finite]
    result.update(
        {
            "all_finite": bool(finite.all()) if values.size else None,
            "finite_count": int(finite.sum()),
            "nan_count": int(nan.sum()),
            "positive_infinity_count": int(posinf.sum()),
            "negative_infinity_count": int(neginf.sum()),
            "minimum": float(finite_values.min()) if finite_values.size else None,
            "maximum": float(finite_values.max()) if finite_values.size else None,
            "mean": float(finite_values.mean()) if finite_values.size else None,
        }
    )
    return result


_SEMANTIC_RULES = {
    "temperature": (
        ("temperature", 1.0, "contains"),
        ("temp", 0.9, "contains"),
        ("thermogram", 0.85, "contains"),
        ("t_", 0.65, "basename prefix"),
    ),
    "heat_flux": (
        ("heat_flux", 1.0, "contains"),
        ("heatflux", 1.0, "contains"),
        ("flux", 0.85, "contains"),
        ("hf_", 0.7, "basename prefix"),
    ),
    "time": (
        ("timestamp", 1.0, "contains"),
        ("time", 0.95, "contains"),
        ("elapsed", 0.8, "contains"),
    ),
    "x_coordinate": (
        ("x_coordinate", 1.0, "contains"),
        ("xcoord", 0.95, "contains"),
        ("x_position", 0.9, "contains"),
        ("x", 0.7, "basename token"),
    ),
    "y_coordinate": (
        ("y_coordinate", 1.0, "contains"),
        ("ycoord", 0.95, "contains"),
        ("y_position", 0.9, "contains"),
        ("y", 0.7, "basename token"),
    ),
}


def infer_semantic_keys(dataset_paths: list[str]) -> dict[str, Any]:
    """Rank transparent name-based semantic candidates without choosing silently."""
    output: dict[str, Any] = {}
    for semantic, rules in _SEMANTIC_RULES.items():
        candidates = []
        for path in dataset_paths:
            lowered = path.lower()
            basename = lowered.rsplit("/", 1)[-1]
            best = None
            for token, confidence, match_type in rules:
                matches = (
                    token in lowered
                    if match_type == "contains"
                    else basename.startswith(token)
                    if match_type == "basename prefix"
                    else basename == token or basename.startswith(token + "_")
                )
                if matches:
                    best = (token, confidence, match_type)
                    break
            if best:
                candidates.append(
                    {
                        "path": path,
                        "rule": f"{best[2]} matches '{best[0]}'",
                        "confidence": best[1],
                    }
                )
        candidates.sort(key=lambda item: (-item["confidence"], item["path"]))
        top = candidates[0]["confidence"] if candidates else None
        tied = [item for item in candidates if item["confidence"] == top]
        output[semantic] = {
            "selected": tied[0]["path"] if len(tied) == 1 else None,
            "certainty": "name-based candidate only" if len(tied) == 1 else (
                "ambiguous" if tied else "no candidate"
            ),
            "alternatives": candidates,
        }
    return output


def inspect_hdf5(path: str | Path, *, sample_limit: int = 4096) -> dict[str, Any]:
    """Inspect an HDF5 file recursively, read-only, using bounded dataset slices."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"HDF5 file does not exist: {file_path}")
    if not isinstance(sample_limit, int) or isinstance(sample_limit, bool) or sample_limit < 1:
        raise ValueError("sample_limit must be a positive integer")
    groups: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    try:
        with h5py.File(file_path, "r") as handle:
            root_attributes = _attributes(handle)

            def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
                full_path = "/" + name
                if isinstance(obj, h5py.Group):
                    groups.append({"path": full_path, "attributes": _attributes(obj)})
                    return
                datasets.append(
                    {
                        "path": full_path,
                        "shape": list(obj.shape),
                        "ndim": obj.ndim,
                        "dtype": str(obj.dtype),
                        "chunks": list(obj.chunks) if obj.chunks is not None else None,
                        "compression": obj.compression,
                        "compression_options": _json_value(obj.compression_opts),
                        "logical_size_bytes": int(obj.size * obj.dtype.itemsize),
                        "attributes": _attributes(obj),
                        "sample": _sample_statistics(obj, sample_limit),
                    }
                )

            handle.visititems(visitor)
    except OSError as exc:
        raise HDF5InspectionError(f"Invalid or unreadable HDF5 file: {file_path}") from exc
    paths = [item["path"] for item in datasets]
    return {
        "schema_version": 1,
        "file": str(file_path.resolve()),
        "open_mode": "read-only",
        "root_attributes": root_attributes,
        "groups": groups,
        "datasets": datasets,
        "dataset_paths": paths,
        "semantic_inference": infer_semantic_keys(paths),
    }


def format_inspection(report: dict[str, Any]) -> str:
    """Format structured inspection output for a readable console report."""
    lines = [f"File: {report['file']}", "Mode: read-only"]
    lines.append(f"Root attributes: {json.dumps(report['root_attributes'], sort_keys=True)}")
    for group in report["groups"]:
        lines.append(f"Group {group['path']} attrs={json.dumps(group['attributes'], sort_keys=True)}")
    for dataset in report["datasets"]:
        lines.append(
            f"Dataset {dataset['path']} shape={tuple(dataset['shape'])} "
            f"dtype={dataset['dtype']} chunks={dataset['chunks']} "
            f"compression={dataset['compression']} logical_bytes={dataset['logical_size_bytes']} "
            f"sampled={dataset['sample']['sampled_elements']}"
        )
    lines.append("Semantic inference: " + json.dumps(report["semantic_inference"], sort_keys=True))
    return "\n".join(lines)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load and fully validate an explicit Task 2 JSON configuration."""
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetPipelineError(f"Cannot read dataset configuration: {config_path}") from exc
    return validate_dataset_config(config)


def _validate_numeric_range(
    config: dict[str, Any], key: str, *, minimum: float, strictly_positive: bool = False
) -> None:
    value = config[key]
    if not isinstance(value, list) or len(value) != 2:
        raise DatasetPipelineError(f"{key} must contain two ordered numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise DatasetPipelineError(f"{key} must contain two ordered numbers")
    low, high = map(float, value)
    if not np.isfinite([low, high]).all() or low > high:
        raise DatasetPipelineError(f"{key} must contain finite ordered values")
    if (strictly_positive and low <= 0.0) or (not strictly_positive and low < minimum):
        qualifier = "positive" if strictly_positive else f"at least {minimum}"
        raise DatasetPipelineError(f"{key} values must be {qualifier}")


def validate_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate every generation choice without running a forward or inverse solve."""
    if not isinstance(config, dict):
        raise DatasetPipelineError("Configuration must be a JSON object")
    required = {
        "schema_version", "random_seed", "output_directory", "grid_shape",
        "num_samples", "source_family_probabilities", "source_count_range",
        "allow_signed_sources", "amplitude_range", "width_range", "size_range",
        "sensor_strategies", "sensor_count_range", "sensor_seeds", "noise_levels",
        "identity_alpha_choices", "smoothness_alpha_choices", "split_rules",
        "ood_source_families", "ood_sensor_strategies", "ood_noise_levels",
        "normalization", "storage_compression", "preview_count",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise DatasetPipelineError(f"Configuration is missing keys: {missing}")
    if config["schema_version"] != DATASET_SCHEMA_VERSION:
        raise DatasetPipelineError(
            f"Unsupported dataset schema version: {config['schema_version']}"
        )
    for key in ("random_seed", "num_samples"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise DatasetPipelineError(f"{key} must be an integer")
    if config["num_samples"] < 1:
        raise DatasetPipelineError("num_samples must be positive")
    if not isinstance(config["allow_signed_sources"], bool):
        raise DatasetPipelineError("allow_signed_sources must be Boolean")
    signed_probability = config.get("signed_probability", 0.0)
    if (
        isinstance(signed_probability, bool)
        or not isinstance(signed_probability, (int, float))
        or not np.isfinite(signed_probability)
        or not 0.0 <= signed_probability <= 1.0
    ):
        raise DatasetPipelineError("signed_probability must be between zero and one")
    if (
        not isinstance(config["grid_shape"], list)
        or len(config["grid_shape"]) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in config["grid_shape"])
        or min(config["grid_shape"]) < 4
    ):
        raise DatasetPipelineError("grid_shape must contain two dimensions of at least 4")
    _validate_numeric_range(config, "amplitude_range", minimum=0.0)
    _validate_numeric_range(config, "width_range", minimum=0.0, strictly_positive=True)
    _validate_numeric_range(config, "size_range", minimum=0.0, strictly_positive=True)
    if (
        not isinstance(config["source_count_range"], list)
        or len(config["source_count_range"]) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in config["source_count_range"])
        or config["source_count_range"][0] < 1
        or config["source_count_range"][0] > config["source_count_range"][1]
    ):
        raise DatasetPipelineError("source_count_range must contain positive ordered integers")
    probabilities = config["source_family_probabilities"]
    if not isinstance(probabilities, dict) or not probabilities:
        raise DatasetPipelineError("source_family_probabilities must be a nonempty object")
    unknown_families = sorted(set(probabilities) - set(SOURCE_FAMILIES))
    if unknown_families:
        raise DatasetPipelineError(f"Unsupported source families: {unknown_families}")
    probability_values = list(probabilities.values())
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in probability_values):
        raise DatasetPipelineError("Source probabilities must be finite and nonnegative")
    if sum(map(float, probability_values)) <= 0:
        raise DatasetPipelineError("Source probabilities must have a positive total")
    strategies = config["sensor_strategies"]
    if not isinstance(strategies, list) or not strategies or len(strategies) != len(set(strategies)):
        raise DatasetPipelineError("Synthetic sensor strategies must be nonempty and unique")
    if "boundary" in strategies:
        raise DatasetPipelineError("boundary is prohibited as a synthetic sensor strategy")
    unsupported = sorted(set(strategies) - set(SYNTHETIC_SENSOR_STRATEGIES))
    if unsupported:
        raise DatasetPipelineError(f"Unsupported synthetic sensor strategies: {unsupported}")
    ood_strategies = config["ood_sensor_strategies"]
    if not isinstance(ood_strategies, list) or not ood_strategies or not set(ood_strategies) <= set(strategies):
        raise DatasetPipelineError("OOD sensor strategies must be available configured strategies")
    if len(ood_strategies) != len(set(ood_strategies)):
        raise DatasetPipelineError("OOD sensor strategies must be unique")
    training_strategies = set(strategies) - set(ood_strategies)
    if not training_strategies:
        raise DatasetPipelineError("At least one non-OOD training sensor strategy is required")
    counts = config["sensor_count_range"]
    interior_capacity = (config["grid_shape"][0] - 2) * (config["grid_shape"][1] - 2)
    if (
        not isinstance(counts, list) or len(counts) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in counts)
        or counts[0] < 1 or counts[0] > counts[1] or counts[1] > interior_capacity
    ):
        raise DatasetPipelineError(
            f"sensor_count_range must be positive, ordered, and at most {interior_capacity} interior nodes"
        )
    seeds = config["sensor_seeds"]
    if not isinstance(seeds, list) or not seeds or any(isinstance(v, bool) or not isinstance(v, int) for v in seeds):
        raise DatasetPipelineError("sensor_seeds must contain integers")
    for key in ("noise_levels", "ood_noise_levels", "identity_alpha_choices", "smoothness_alpha_choices"):
        values = config[key]
        if not isinstance(values, list) or not values:
            raise DatasetPipelineError(f"{key} must be nonempty")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values):
            raise DatasetPipelineError(f"{key} must contain finite numbers")
    if any(float(v) < 0 for v in config["noise_levels"]):
        raise DatasetPipelineError("noise_levels must be nonnegative")
    if any(float(v) <= 0 for key in ("identity_alpha_choices", "smoothness_alpha_choices") for v in config[key]):
        raise DatasetPipelineError("All alpha choices must be positive")
    if not set(map(float, config["ood_noise_levels"])) <= set(map(float, config["noise_levels"])):
        raise DatasetPipelineError("OOD noise levels must be available configured noise levels")
    ood_families = config["ood_source_families"]
    if not isinstance(ood_families, list) or not ood_families or not set(ood_families) <= set(probabilities):
        raise DatasetPipelineError("OOD source families must be available configured families")
    rules = config["split_rules"]
    if "counts" in rules:
        split_counts = rules["counts"]
        if set(split_counts) != set(SPLIT_ROLES) or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in split_counts.values()):
            raise DatasetPipelineError("Every required split count must be a positive integer")
        if sum(split_counts.values()) != config["num_samples"]:
            raise DatasetPipelineError("Explicit split counts must sum to num_samples")
    elif "fractions" in rules:
        fractions = rules["fractions"]
        if set(fractions) != set(SPLIT_ROLES) or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in fractions.values()):
            raise DatasetPipelineError("Every required split fraction must be positive")
        if not np.isclose(sum(map(float, fractions.values())), 1.0):
            raise DatasetPipelineError("Split fractions must sum to one")
        planned_counts = {role: int(np.floor(config["num_samples"] * float(fractions[role]))) for role in SPLIT_ROLES}
        if any(value < 1 for value in planned_counts.values()):
            raise DatasetPipelineError("Every required role must receive at least one planned sample")
    else:
        raise DatasetPipelineError("split_rules must contain counts or fractions")
    normalization = config["normalization"]
    if not isinstance(normalization, dict) or not isinstance(normalization.get("enabled"), bool):
        raise DatasetPipelineError("normalization.enabled must be Boolean")
    if normalization["enabled"] and normalization.get("method") != "global_standard":
        raise DatasetPipelineError("Enabled normalization method must be global_standard")
    if config["storage_compression"] not in {"gzip", "lzf", None}:
        raise DatasetPipelineError("storage_compression must be gzip, lzf, or null")
    return config


SOURCE_FAMILIES = (
    "one_gaussian", "multiple_gaussians", "elliptical_hotspot", "rectangle",
    "circular_compact", "elongated_source", "sharp_edged",
    "overlapping_hotspots", "irregular_composite",
)


def construct_source(
    grid: Grid2D,
    family: str,
    *,
    seed: int,
    amplitude_range: tuple[float, float] = (0.5, 2.0),
    width_range: tuple[float, float] = (0.05, 0.16),
    size_range: tuple[float, float] = (0.08, 0.28),
    source_count_range: tuple[int, int] = (2, 4),
    signed: bool = False,
    allow_signed: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct a deterministic research source while reusing classical generators."""
    if family not in SOURCE_FAMILIES:
        raise ValueError(f"Unknown source family: {family}")
    if signed and not allow_signed:
        raise ValueError("Signed sources require allow_signed=True")
    rng = np.random.default_rng(seed)
    amplitude = float(rng.uniform(*amplitude_range))
    width = float(rng.uniform(*width_range))
    size = float(rng.uniform(*size_range))
    cx, cy = rng.uniform(0.28, 0.72, size=2)
    x = grid.X
    y = grid.Y
    parameters: dict[str, Any] = {
        "family": family, "amplitude": amplitude, "width": width,
        "size": size, "center": [float(cx), float(cy)], "signed": signed,
    }
    if family == "one_gaussian":
        source = gaussian_source(grid, center=(cx, cy), amplitude=amplitude, sigma=width)
    elif family == "multiple_gaussians":
        count = int(rng.integers(source_count_range[0], source_count_range[1] + 1))
        hotspots = []
        source = np.zeros(grid.shape, dtype=np.float64)
        for _ in range(count):
            hotspot = {
                "center": [float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85))],
                "amplitude": float(rng.uniform(*amplitude_range)),
                "sigma": float(rng.uniform(*width_range)),
            }
            hotspots.append(hotspot)
            source += gaussian_source(
                grid,
                center=tuple(hotspot["center"]),
                amplitude=hotspot["amplitude"],
                sigma=hotspot["sigma"],
            )
        parameters.update({
            "count": count,
            "generator": "sum_of_thermoreconlab_gaussian_source",
            "hotspots": hotspots,
        })
    elif family in {"elliptical_hotspot", "elongated_source"}:
        aspect = 2.0 if family == "elliptical_hotspot" else 4.0
        theta = float(rng.uniform(0.0, np.pi))
        xr = np.cos(theta) * (x - cx) + np.sin(theta) * (y - cy)
        yr = -np.sin(theta) * (x - cx) + np.cos(theta) * (y - cy)
        source = amplitude * np.exp(-0.5 * ((xr / (width * aspect)) ** 2 + (yr / width) ** 2))
        parameters.update({"aspect_ratio": aspect, "angle_radians": theta})
    elif family == "rectangle":
        half_x, half_y = size, size * float(rng.uniform(0.45, 0.9))
        source = amplitude * ((np.abs(x - cx) <= half_x) & (np.abs(y - cy) <= half_y))
        parameters["half_sizes"] = [half_x, half_y]
    elif family == "circular_compact":
        source = amplitude * (((x - cx) ** 2 + (y - cy) ** 2) <= size**2)
        parameters["radius"] = size
    elif family == "sharp_edged":
        source = amplitude * ((np.abs(x - cx) / size + np.abs(y - cy) / size) <= 1.0)
        parameters["diamond_half_diagonal"] = size
    elif family == "overlapping_hotspots":
        offset = width * 0.7
        source = gaussian_source(grid, center=(cx - offset, cy), amplitude=amplitude, sigma=width)
        source += gaussian_source(grid, center=(cx + offset, cy), amplitude=0.8 * amplitude, sigma=width)
        parameters["offset"] = offset
    else:
        circle = ((x - cx) ** 2 + (y - cy) ** 2) <= size**2
        bar = (np.abs(x - (cx + size * 0.6)) <= size * 0.7) & (np.abs(y - cy) <= size * 0.25)
        blob = np.exp(-0.5 * (((x - (cx - size)) / width) ** 2 + ((y - (cy + size * 0.5)) / width) ** 2))
        source = amplitude * (circle.astype(float) + 0.65 * bar.astype(float) + 0.5 * blob)
        parameters["components"] = ["circle", "bar", "gaussian"]
    source = np.asarray(source, dtype=np.float64)
    if signed:
        sign_mask = np.where(x < cx, -1.0, 1.0)
        source = source * sign_mask
        parameters["sign_rule"] = "negative left of generated center"
    return source, parameters


def _sensor_indices(
    grid: Grid2D, strategy: str, count: int, seed: int, *, allow_boundary: bool = False
) -> np.ndarray:
    if strategy == "regular_grid":
        return regular_grid_sensors(grid, count)
    if strategy == "random":
        return random_sensors(grid, count, seed=seed)
    if strategy == "boundary":
        if not allow_boundary:
            raise ValueError("boundary is prohibited as a synthetic sensor strategy")
        return boundary_sensors(grid, count)
    if strategy == "center_focused":
        return center_focused_sensors(grid, count, seed=seed)
    raise ValueError(f"Unknown sensor strategy: {strategy}")


def _stable_sample_id(metadata: dict[str, Any]) -> str:
    return "syn-" + hashlib.sha256(_canonical_json(metadata)).hexdigest()[:24]


def source_valid_mask(grid_shape: tuple[int, int] | list[int]) -> np.ndarray:
    """Return the reusable interior-node target mask for Dirichlet source fields."""
    shape = tuple(map(int, grid_shape))
    mask = np.zeros(shape, dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask


SPLIT_ROLES = (
    "train", "validation", "test_id", "test_ood_shape",
    "test_ood_sensor", "test_ood_noise",
)


def plan_synthetic_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Plan deterministic, seed-disjoint split membership before generation."""
    validate_dataset_config(config)
    count = int(config["num_samples"])
    weights = config["source_family_probabilities"]
    ood_families = set(config["ood_source_families"])
    ood_sensors = set(config["ood_sensor_strategies"])
    ood_noise = {float(value) for value in config["ood_noise_levels"]}
    train_families = [name for name in weights if name not in ood_families and weights[name] > 0]
    id_sensors = [name for name in config["sensor_strategies"] if name not in ood_sensors]
    id_noise = [float(value) for value in config["noise_levels"] if float(value) not in ood_noise]
    if not train_families or not id_sensors or not id_noise:
        raise DatasetPipelineError("Training requires non-OOD source, sensor, and noise choices")
    requested = config["split_rules"].get("counts")
    if requested:
        roles = [role for role in SPLIT_ROLES for _ in range(int(requested.get(role, 0)))]
        if len(roles) != count:
            raise DatasetPipelineError("Explicit split counts must sum to num_samples")
    else:
        fractions = config["split_rules"].get("fractions")
        if not fractions or set(fractions) != set(SPLIT_ROLES):
            raise DatasetPipelineError("split_rules must provide counts or all split fractions")
        fraction_total = sum(float(fractions[role]) for role in SPLIT_ROLES)
        if not np.isclose(fraction_total, 1.0):
            raise DatasetPipelineError("Split fractions must sum to one")
        raw_counts = {role: count * float(fractions[role]) for role in SPLIT_ROLES}
        split_counts = {role: int(np.floor(raw_counts[role])) for role in SPLIT_ROLES}
        remainder = count - sum(split_counts.values())
        order = sorted(SPLIT_ROLES, key=lambda role: (-(raw_counts[role] - split_counts[role]), role))
        for role in order[:remainder]:
            split_counts[role] += 1
        roles = [role for role in SPLIT_ROLES for _ in range(split_counts[role])]
    base = int(config["random_seed"])
    sensor_seed_base = int(config["sensor_seeds"][0])
    plans = []
    family_probability = np.array([float(weights[name]) for name in train_families], dtype=float)
    family_probability /= family_probability.sum()
    for index, role in enumerate(roles):
        rng = np.random.default_rng(base + index * 1009)
        family = (
            sorted(ood_families)[index % len(ood_families)]
            if role == "test_ood_shape" else str(rng.choice(train_families, p=family_probability))
        )
        strategy = (
            sorted(ood_sensors)[index % len(ood_sensors)]
            if role == "test_ood_sensor" else id_sensors[index % len(id_sensors)]
        )
        noise = (
            sorted(ood_noise)[index % len(ood_noise)]
            if role == "test_ood_noise" else id_noise[index % len(id_noise)]
        )
        reason = {
            "train": "training in-distribution source, sensor, and noise",
            "validation": "held-out source seed for validation",
            "test_id": "held-out source seed for in-distribution testing",
            "test_ood_shape": "source family reserved from training",
            "test_ood_sensor": "sensor strategy reserved from training",
            "test_ood_noise": "noise level reserved from training",
        }[role]
        low, high = map(int, config["sensor_count_range"])
        sensor_count = int(rng.integers(low, high + 1))
        identity_alpha = float(
            config["identity_alpha_choices"][index % len(config["identity_alpha_choices"])]
        )
        smoothness_alpha = float(
            config["smoothness_alpha_choices"][index % len(config["smoothness_alpha_choices"])]
        )
        plans.append({
            "ordinal": index, "split": role, "split_reason": reason,
            "source_family": family, "sensor_strategy": strategy, "noise_level": noise,
            "generation_seed": base + index, "source_seed": base + 100_000 + index,
            "sensor_seed": sensor_seed_base + index, "noise_seed": base + 300_000 + index,
            "sensor_count": sensor_count, "identity_alpha": identity_alpha,
            "smoothness_alpha": smoothness_alpha,
        })
    return plans


def generate_synthetic_sample(config: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Generate one exact supervised pair and two classical baselines."""
    grid = Grid2D(*map(int, config["grid_shape"]))
    rng = np.random.default_rng(plan["generation_seed"])
    signed = bool(config["allow_signed_sources"] and rng.random() < float(config.get("signed_probability", 0.0)))
    constructed_source, source_configuration = construct_source(
        grid, plan["source_family"], seed=plan["source_seed"],
        amplitude_range=tuple(config["amplitude_range"]),
        width_range=tuple(config["width_range"]), size_range=tuple(config["size_range"]),
        source_count_range=tuple(config["source_count_range"]), signed=signed,
        allow_signed=bool(config["allow_signed_sources"]),
    )
    source = constructed_source.copy()
    source[[0, -1], :] = 0.0
    source[:, [0, -1]] = 0.0
    temperature = solve_forward(source, grid)
    sensor_count = int(plan["sensor_count"])
    indices = _sensor_indices(grid, plan["sensor_strategy"], sensor_count, plan["sensor_seed"])
    clean = sample_field(temperature, indices, grid)
    measured = add_gaussian_noise(
        clean, noise_level=float(plan["noise_level"]), seed=plan["noise_seed"], relative=True
    )
    sensor_data = SensorData(indices=indices.copy(), values=measured.copy())
    mask = np.zeros(grid.shape, dtype=bool)
    sparse = np.zeros(grid.shape, dtype=float)
    mask[indices[:, 0], indices[:, 1]] = True
    sparse[indices[:, 0], indices[:, 1]] = measured
    identity_alpha = float(plan["identity_alpha"])
    smoothness_alpha = float(plan["smoothness_alpha"])
    identity = reconstruct_tikhonov(sensor_data, grid, alpha=identity_alpha, regularization="identity").source.copy()
    smoothness = reconstruct_smooth_tikhonov(
        sensor_data, grid, alpha=smoothness_alpha, nonnegative=not signed
    ).source.copy()
    identity[[0, -1], :] = 0.0
    identity[:, [0, -1]] = 0.0
    smoothness[[0, -1], :] = 0.0
    smoothness[:, [0, -1]] = 0.0
    source_target_domain = "interior_nodes"
    boundary_source_policy = "zeroed_to_match_homogeneous_dirichlet_forward_model"
    observation_identifier = "thermoreconlab.sensor_indices:C-order:v1"
    immutable = {
        "schema_version": DATASET_SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
        "grid_shape": list(grid.shape), "source_configuration": source_configuration,
        "source_seed": plan["source_seed"], "sensor_seed": plan["sensor_seed"],
        "signed_source": signed, "source_target_domain": source_target_domain,
        "boundary_source_policy": boundary_source_policy,
        "sensor_strategy": plan["sensor_strategy"], "noise_level": plan["noise_level"],
        "sensor_count": sensor_count, "noise_seed": plan["noise_seed"],
        "identity_alpha": identity_alpha, "smoothness_alpha": smoothness_alpha,
        "observation_operator_identifier": observation_identifier,
    }
    return {
        "sample_id": _stable_sample_id(immutable), "task_type": "synthetic_source",
        "split": plan["split"], "split_reason": plan["split_reason"],
        "grid_shape": list(grid.shape), "true_source": source.copy(),
        "full_temperature": temperature.copy(), "sparse_temperature": sparse,
        "sensor_mask": mask, "sensor_indices": indices.copy(), "sensor_count": len(indices),
        "measured_temperatures": measured.copy(), "identity_reconstruction": identity.copy(),
        "smoothness_reconstruction": smoothness.copy(), "source_family": plan["source_family"],
        "source_configuration": source_configuration,
        "source_target_domain": source_target_domain,
        "boundary_source_policy": boundary_source_policy,
        "sensor_configuration": {"strategy": plan["sensor_strategy"], "count": len(indices), "simulated": True},
        "noise_configuration": {"relative_std": float(plan["noise_level"]), "realization_seed": plan["noise_seed"]},
        "identity_alpha": identity_alpha, "smoothness_alpha": smoothness_alpha,
        "generation_seed": plan["generation_seed"], "source_seed": plan["source_seed"],
        "sensor_seed": plan["sensor_seed"], "noise_seed": plan["noise_seed"],
        "observation_operator_identifier": observation_identifier,
    }


NORMALIZABLE_FIELDS = (
    "true_source", "full_temperature", "sparse_temperature",
    "identity_reconstruction", "smoothness_reconstruction",
)


def fit_training_normalization(
    samples: list[dict[str, Any]], *, fields: tuple[str, ...] = NORMALIZABLE_FIELDS
) -> dict[str, Any]:
    """Fit global standardization using train samples and no other split."""
    training = [sample for sample in samples if sample["split"] == "train"]
    if not training:
        raise DatasetPipelineError("Cannot fit normalization without training samples")
    statistics: dict[str, Any] = {}
    for field in fields:
        total = 0
        value_sum = 0.0
        square_sum = 0.0
        for sample in training:
            values = np.asarray(sample[field], dtype=np.float64)
            if not np.isfinite(values).all():
                raise DatasetPipelineError(f"Non-finite training values in {field}")
            total += values.size
            value_sum += float(values.sum(dtype=np.float64))
            square_sum += float(np.square(values).sum(dtype=np.float64))
        mean = value_sum / total
        variance = max(square_sum / total - mean * mean, 0.0)
        raw_std = variance**0.5
        statistics[field] = {
            "mean": mean, "standard_deviation": raw_std,
            "scale": raw_std if raw_std > 0.0 else 1.0,
            "zero_variance": raw_std == 0.0, "element_count": total,
        }
    return {
        "schema_version": DATASET_SCHEMA_VERSION, "method": "global_standard",
        "fitted_split": "train", "sample_ids": [sample["sample_id"] for sample in training],
        "statistics": statistics,
    }


def normalize_array(values: np.ndarray, statistics: dict[str, Any]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - statistics["mean"]) / statistics["scale"]


def inverse_normalize_array(values: np.ndarray, statistics: dict[str, Any]) -> np.ndarray:
    return np.asarray(values, dtype=float) * statistics["scale"] + statistics["mean"]


def _manifest_content_hash(manifest: dict[str, Any]) -> str:
    content = dict(manifest)
    content.pop("manifest_content_sha256", None)
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def build_synthetic_dataset(
    config: dict[str, Any], *, output_directory: str | Path | None = None
) -> dict[str, Any]:
    """Build an integrity-checked dataset, publishing its manifest last."""
    validate_dataset_config(config)
    plans = plan_synthetic_samples(config)
    samples = [generate_synthetic_sample(config, plan) for plan in plans]
    output = Path(output_directory or config["output_directory"])
    names = ("synthetic_dataset.h5", "configuration.json", "normalization.json", "manifest.json")
    finals = {name: output / name for name in names}
    parts = {name: output / f"{name}.part" for name in names}
    existing = [str(path) for path in (*finals.values(), *parts.values()) if path.exists()]
    if existing:
        raise DatasetPipelineError(f"Dataset output already exists or is partial: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    config_snapshot = json.loads(json.dumps(config))
    config_hash = hashlib.sha256(_canonical_json(config_snapshot)).hexdigest()
    compression = config["storage_compression"]
    count = len(samples)
    shape = tuple(config["grid_shape"])
    maximum_sensors = max(sample["sensor_count"] for sample in samples)
    try:
        with h5py.File(parts["synthetic_dataset.h5"], "w") as handle:
            handle.attrs.update({
                "schema_version": DATASET_SCHEMA_VERSION, "complete": False,
                "sample_count": count, "configuration_sha256": config_hash,
            })
            for field in NORMALIZABLE_FIELDS:
                data = np.stack([sample[field] for sample in samples])
                handle.create_dataset(field, data=data, chunks=(1, *shape), compression=compression)
            handle.create_dataset(
                "sensor_mask", data=np.stack([sample["sensor_mask"] for sample in samples]),
                chunks=(1, *shape), compression=compression,
            )
            padded_indices = np.full((count, maximum_sensors, 2), -1, dtype=np.int64)
            padded_values = np.full((count, maximum_sensors), np.nan, dtype=np.float64)
            for index, sample in enumerate(samples):
                n = sample["sensor_count"]
                padded_indices[index, :n] = sample["sensor_indices"]
                padded_values[index, :n] = sample["measured_temperatures"]
            handle.create_dataset("sensor_indices", data=padded_indices, compression=compression)
            handle.create_dataset("measured_temperatures", data=padded_values, compression=compression)
            handle.create_dataset("sensor_count", data=[s["sensor_count"] for s in samples])
            handle.create_dataset("source_valid_mask", data=source_valid_mask(shape))
            handle.attrs["complete"] = True
            handle.flush()
    except Exception:
        raise
    metadata_keys = [
        "sample_id", "task_type", "split", "split_reason", "grid_shape", "sensor_count",
        "source_family", "source_configuration", "sensor_configuration", "noise_configuration",
        "identity_alpha", "smoothness_alpha", "generation_seed", "source_seed", "sensor_seed",
        "noise_seed", "observation_operator_identifier", "source_target_domain",
        "boundary_source_policy",
    ]
    normalization = (
        fit_training_normalization(samples)
        if config["normalization"].get("enabled", True)
        else {"schema_version": DATASET_SCHEMA_VERSION, "method": "none", "sample_ids": []}
    )
    normalization_hash = hashlib.sha256(_canonical_json(normalization)).hexdigest()
    parts["configuration.json"].write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parts["normalization.json"].write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dataset_hash = _sha256_file(parts["synthetic_dataset.h5"])
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_file": finals["synthetic_dataset.h5"].name,
        "configuration_file": finals["configuration.json"].name,
        "normalization_file": finals["normalization.json"].name,
        "sample_count": count, "dataset_sha256": dataset_hash,
        "configuration_sha256": config_hash, "normalization_sha256": normalization_hash,
        "samples": [
            {"storage_index": i, **{key: sample[key] for key in metadata_keys}}
            for i, sample in enumerate(samples)
        ],
    }
    manifest["manifest_content_sha256"] = _manifest_content_hash(manifest)
    parts["manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in ("synthetic_dataset.h5", "configuration.json", "normalization.json"):
        parts[name].replace(finals[name])
    parts["manifest.json"].replace(finals["manifest.json"])
    return {
        "output_directory": str(output.resolve()),
        "dataset_path": str(finals["synthetic_dataset.h5"].resolve()),
        "manifest_path": str(finals["manifest.json"].resolve()),
        "normalization_path": str(finals["normalization.json"].resolve()),
        "sample_count": count, "split_counts": {role: sum(s["split"] == role for s in samples) for role in SPLIT_ROLES},
        "dataset_bytes": finals["synthetic_dataset.h5"].stat().st_size,
        "dataset_sha256": dataset_hash, "configuration_sha256": config_hash,
        "normalization_sha256": normalization_hash,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
    }


class SyntheticDatasetReader:
    """Lazy read-only indexed access to a generated synthetic dataset."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.dataset_path = self.directory / "synthetic_dataset.h5"
        self.configuration_path = self.directory / "configuration.json"
        self.normalization_path = self.directory / "normalization.json"
        for path in (self.dataset_path, self.configuration_path, self.normalization_path, self.manifest_path):
            if not path.is_file():
                raise DatasetPipelineError(f"Required synthetic dataset artifact is missing: {path.name}")
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.configuration = json.loads(self.configuration_path.read_text(encoding="utf-8"))
            self.normalization = json.loads(self.normalization_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetPipelineError("Missing or invalid synthetic manifest") from exc
        if self.manifest.get("manifest_content_sha256") != _manifest_content_hash(self.manifest):
            raise DatasetPipelineError("Synthetic manifest checksum is invalid")
        if self.manifest.get("sample_count") != len(self.manifest.get("samples", [])):
            raise DatasetPipelineError("Synthetic manifest sample count is invalid")
        if self.manifest.get("configuration_sha256") != hashlib.sha256(_canonical_json(self.configuration)).hexdigest():
            raise DatasetPipelineError("Synthetic configuration checksum is invalid")
        if self.manifest.get("normalization_sha256") != hashlib.sha256(_canonical_json(self.normalization)).hexdigest():
            raise DatasetPipelineError("Synthetic normalization checksum is invalid")
        if self.manifest.get("dataset_sha256") != _sha256_file(self.dataset_path):
            raise DatasetPipelineError("Synthetic HDF5 checksum is invalid")
        for label, payload in (("manifest", self.manifest), ("configuration", self.configuration), ("normalization", self.normalization)):
            if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
                raise DatasetPipelineError(f"Synthetic {label} schema version is invalid")
        self._handle: h5py.File | None = None

    def open(self) -> "SyntheticDatasetReader":
        if self._handle is None:
            try:
                self._handle = h5py.File(self.dataset_path, "r")
            except OSError as exc:
                raise DatasetPipelineError("Missing or corrupt synthetic HDF5 dataset") from exc
            if not bool(self._handle.attrs.get("complete", False)):
                self.close()
                raise DatasetPipelineError("Synthetic HDF5 dataset is incomplete")
            if int(self._handle.attrs.get("sample_count", -1)) != len(self.manifest["samples"]):
                self.close()
                raise DatasetPipelineError("Synthetic sample count does not match manifest")
            if int(self._handle.attrs.get("schema_version", -1)) != DATASET_SCHEMA_VERSION:
                self.close()
                raise DatasetPipelineError("Synthetic HDF5 schema version is invalid")
            if self._handle.attrs.get("configuration_sha256") != self.manifest["configuration_sha256"]:
                self.close()
                raise DatasetPipelineError("Synthetic HDF5 configuration checksum is invalid")
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "SyntheticDatasetReader":
        return self.open()

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.manifest["samples"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.open()
        assert self._handle is not None
        metadata = dict(self.manifest["samples"][index])
        n = int(metadata["sensor_count"])
        stored_n = int(self._handle["sensor_count"][index])
        if n <= 0 or stored_n != n:
            raise DatasetPipelineError("Corrupt sensor_count: HDF5 and manifest disagree or count is nonpositive")
        sample = dict(metadata)
        for field in (*NORMALIZABLE_FIELDS, "sensor_mask"):
            sample[field] = np.array(self._handle[field][index], copy=True)
        sample["sensor_indices"] = np.array(self._handle["sensor_indices"][index, :n], copy=True)
        sample["measured_temperatures"] = np.array(self._handle["measured_temperatures"][index, :n], copy=True)
        sample["source_valid_mask"] = np.array(self._handle["source_valid_mask"][...], copy=True)
        for field in (*NORMALIZABLE_FIELDS, "measured_temperatures"):
            if not np.isfinite(sample[field]).all():
                raise DatasetPipelineError(f"Corrupt non-finite sample field: {field}")
        shape = tuple(metadata["grid_shape"])
        for field in (*NORMALIZABLE_FIELDS, "sensor_mask", "source_valid_mask"):
            if sample[field].shape != shape:
                raise DatasetPipelineError(f"Corrupt {field}: expected shape {shape}, found {sample[field].shape}")
        indices = sample["sensor_indices"]
        if indices.shape != (n, 2):
            raise DatasetPipelineError(f"Corrupt sensor_indices: expected {(n, 2)}, found {indices.shape}")
        if np.any(indices < 0) or np.any(indices[:, 0] >= shape[0]) or np.any(indices[:, 1] >= shape[1]):
            raise DatasetPipelineError("Corrupt sensor_indices: index is out of bounds")
        if len(np.unique(indices, axis=0)) != n:
            raise DatasetPipelineError("Corrupt sensor_indices: indices are not unique")
        expected_mask = np.zeros(shape, dtype=bool)
        expected_mask[indices[:, 0], indices[:, 1]] = True
        if not np.array_equal(sample["sensor_mask"], expected_mask):
            raise DatasetPipelineError("Corrupt sensor_mask: mask does not exactly match indices")
        if not np.array_equal(sample["source_valid_mask"], source_valid_mask(shape)):
            raise DatasetPipelineError("Corrupt source_valid_mask: mask does not match grid interior")
        sparse = sample["sparse_temperature"]
        if not np.array_equal(
            sparse[indices[:, 0], indices[:, 1]], sample["measured_temperatures"]
        ):
            raise DatasetPipelineError("Corrupt sparse_temperature: sensor values disagree with measurements")
        if not np.all(sparse[~expected_mask] == 0.0):
            raise DatasetPipelineError("Corrupt sparse_temperature: values outside sensor mask are nonzero")
        return sample

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state


def _external_frame_map(handle: h5py.File, pattern: re.Pattern[str], label: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    unexpected = []
    for key in handle.keys():
        match = pattern.fullmatch(key)
        if match is None:
            unexpected.append(key)
            continue
        frame = int(match.group(1))
        if frame in mapping:
            raise DatasetPipelineError(f"Duplicate {label} frame index: {frame}")
        mapping[frame] = key
    if unexpected:
        raise DatasetPipelineError(f"Unexpected {label} HDF5 keys: {unexpected[:5]}")
    if not mapping:
        raise DatasetPipelineError(f"No {label} frame keys found")
    expected = set(range(min(mapping), max(mapping) + 1))
    missing = sorted(expected - mapping.keys())
    if missing:
        raise DatasetPipelineError(f"Missing {label} frame indices: {missing[:10]}")
    return mapping


def build_external_manifest(
    temperature_path: str | Path,
    heat_flux_path: str | Path,
    output_path: str | Path,
    *,
    experiment_id: str,
    plate_id: str,
    usage_role: str = "external_audit",
    calculate_checksums: bool = True,
) -> dict[str, Any]:
    """Validate and record a read-only external temperature/reference pair."""
    temperature_file = Path(temperature_path)
    flux_file = Path(heat_flux_path)
    for path in (temperature_file, flux_file):
        if not path.is_file():
            raise FileNotFoundError(f"External HDF5 file does not exist: {path}")
    try:
        with h5py.File(temperature_file, "r") as temperature, h5py.File(flux_file, "r") as flux:
            temperature_frames = _external_frame_map(temperature, EXTERNAL_TEMPERATURE_PATTERN, "temperature")
            flux_frames = _external_frame_map(flux, EXTERNAL_FLUX_PATTERN, "heat-flux")
            if set(temperature_frames) != set(flux_frames):
                raise DatasetPipelineError("Temperature and heat-flux frame-index sets differ")
            frame_indices = sorted(temperature_frames)
            shapes = set()
            temperature_dtypes = set()
            flux_dtypes = set()
            for frame in frame_indices:
                t_dataset = temperature[temperature_frames[frame]]
                f_dataset = flux[flux_frames[frame]]
                if t_dataset.shape != f_dataset.shape:
                    raise DatasetPipelineError(f"External frame {frame} shapes differ")
                shapes.add(tuple(t_dataset.shape))
                temperature_dtypes.add(str(t_dataset.dtype))
                flux_dtypes.add(str(f_dataset.dtype))
            if len(shapes) != 1:
                raise DatasetPipelineError("External frames do not have one consistent shape")
    except OSError as exc:
        raise DatasetPipelineError("External pair contains invalid HDF5") from exc
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "task_type": "external_heat_flux", "usage_role": usage_role,
        "experiment_id": experiment_id, "plate_id": plate_id,
        "temperature_path": str(temperature_file.resolve()),
        "heat_flux_path": str(flux_file.resolve()),
        "temperature_size_bytes": temperature_file.stat().st_size,
        "heat_flux_size_bytes": flux_file.stat().st_size,
        "temperature_sha256": _sha256_file(temperature_file) if calculate_checksums else None,
        "heat_flux_sha256": _sha256_file(flux_file) if calculate_checksums else None,
        "temperature_key_pattern": EXTERNAL_TEMPERATURE_PATTERN.pattern,
        "heat_flux_key_pattern": EXTERNAL_FLUX_PATTERN.pattern,
        "frame_indices": frame_indices, "frame_count": len(frame_indices),
        "frame_shape": list(next(iter(shapes))),
        "temperature_dtypes": sorted(temperature_dtypes), "heat_flux_dtypes": sorted(flux_dtypes),
        "time_semantics": "ordered_frame_index_only", "time_values": None,
        "time_units": "unresolved", "temperature_units": "unresolved",
        "heat_flux_units": "unresolved", "x_coordinates": None, "y_coordinates": None,
        "coordinate_units": "unresolved", "physical_orientation": "unresolved",
        "gauge_registration": "unresolved",
        "target_provenance": "HFITS-derived external reference heat-flux estimate",
        "independent_ground_truth": False, "classical_q_target": False,
        "sparse_sampling_provenance": "simulated sparse samples from a real full-field temperature frame",
    }
    manifest["manifest_content_sha256"] = _manifest_content_hash(manifest)
    _atomic_json(Path(output_path), manifest)
    return manifest


def assign_experiment_splits(
    records: list[dict[str, Any]],
    *,
    seed: int,
    split_names: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, str]:
    """Assign whole experiments deterministically; frames and plates never cross splits."""
    experiments = sorted({str(record["experiment_id"]) for record in records})
    if len(split_names) > 1 and len(experiments) < len(split_names):
        raise DatasetPipelineError(
            f"Cannot create {len(split_names)} experiment-level splits from {len(experiments)} experiment(s)"
        )
    ordered = sorted(
        experiments,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    return {experiment: split_names[index % len(split_names)] for index, experiment in enumerate(ordered)}


def _window_indices(center: int, count: int, window_size: int, policy: str) -> list[int]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(center, bool) or not isinstance(center, int) or not 0 <= center < count:
        raise ValueError("center must be an integer inside the available frame range")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if policy not in {"reject", "edge-repeat", "reflect"}:
        raise ValueError("boundary_policy must be reject, edge-repeat, or reflect")
    half = window_size // 2
    requested = list(range(center - half, center + half + 1))
    if policy == "reject":
        if min(requested) < 0 or max(requested) >= count:
            raise DatasetPipelineError("Temporal window crosses the frame boundary")
        return requested
    if count == 1:
        return [0] * window_size
    result = []
    for index in requested:
        if policy == "edge-repeat":
            result.append(min(max(index, 0), count - 1))
        else:
            reflected = index
            while reflected < 0 or reflected >= count:
                reflected = -reflected if reflected < 0 else 2 * count - 2 - reflected
            result.append(reflected)
    return result


class ExternalWindowReader:
    """Lazy, read-only temporal access to an external experimental pair."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        window_size: int,
        boundary_policy: str,
        sensor_strategy: str,
        sensor_count: int,
        sensor_seed: int,
    ):
        self.manifest_path = Path(manifest_path)
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetPipelineError("Missing or invalid external manifest") from exc
        if self.manifest.get("manifest_content_sha256") != _manifest_content_hash(self.manifest):
            raise DatasetPipelineError("External manifest checksum is invalid")
        if self.manifest.get("task_type") != "external_heat_flux":
            raise DatasetPipelineError("External manifest task_type is invalid")
        if self.manifest.get("usage_role") != "external_audit":
            raise DatasetPipelineError("The available PR pair must remain external_audit")
        if self.manifest.get("target_provenance") != "HFITS-derived external reference heat-flux estimate":
            raise DatasetPipelineError("External target provenance is invalid")
        if self.manifest.get("independent_ground_truth") is not False or self.manifest.get("classical_q_target") is not False:
            raise DatasetPipelineError("External scientific target flags are invalid")
        for path_key, size_key in (
            ("temperature_path", "temperature_size_bytes"),
            ("heat_flux_path", "heat_flux_size_bytes"),
        ):
            path = Path(self.manifest[path_key])
            if not path.is_file() or path.stat().st_size != self.manifest.get(size_key):
                raise DatasetPipelineError(f"External file is missing or size changed: {path_key}")
        frame_count = self.manifest["frame_count"]
        _window_indices(0, frame_count, 1, boundary_policy)
        _window_indices(0, frame_count, window_size, "edge-repeat")
        self.window_size = window_size
        self.boundary_policy = boundary_policy
        self.sensor_strategy = sensor_strategy
        self.sensor_count = sensor_count
        self.sensor_seed = sensor_seed
        self._temperature: h5py.File | None = None
        self._flux: h5py.File | None = None

    def open(self) -> "ExternalWindowReader":
        if self._temperature is None:
            self._temperature = h5py.File(self.manifest["temperature_path"], "r")
            self._flux = h5py.File(self.manifest["heat_flux_path"], "r")
        return self

    def close(self) -> None:
        if self._temperature is not None:
            self._temperature.close()
            self._temperature = None
        if self._flux is not None:
            self._flux.close()
            self._flux = None

    def __enter__(self) -> "ExternalWindowReader":
        return self.open()

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_temperature"] = None
        state["_flux"] = None
        return state

    def get(self, central_frame_index: int) -> dict[str, Any]:
        frames = self.manifest["frame_indices"]
        if central_frame_index not in frames:
            raise DatasetPipelineError(f"External frame is missing: {central_frame_index}")
        positions = _window_indices(frames.index(central_frame_index), len(frames), self.window_size, self.boundary_policy)
        selected_frames = [frames[position] for position in positions]
        self.open()
        assert self._temperature is not None and self._flux is not None
        temperature_arrays = []
        for frame in selected_frames:
            key = f"surface_temperature_batch0_frame{frame:06d}"
            if key not in self._temperature:
                raise DatasetPipelineError(f"External temperature frame is missing: {frame}")
            temperature_arrays.append(np.array(self._temperature[key][...], copy=True))
        flux_key = f"estimated_flux_batch0_frame{central_frame_index:06d}"
        if flux_key not in self._flux:
            raise DatasetPipelineError(f"External heat-flux frame is missing: {central_frame_index}")
        flux = np.array(self._flux[flux_key][...], copy=True)
        window = np.stack(temperature_arrays)
        if not np.isfinite(window).all() or not np.isfinite(flux).all():
            raise DatasetPipelineError("External window contains non-finite values")
        central = np.array(temperature_arrays[self.window_size // 2], copy=True)
        height, width = central.shape
        grid = Grid2D(height, width)
        stable_seed = int.from_bytes(
            hashlib.sha256(
                f"{self.manifest['experiment_id']}:{self.manifest['plate_id']}:{central_frame_index}:{self.sensor_seed}".encode()
            ).digest()[:8], "big"
        )
        indices = _sensor_indices(
            grid, self.sensor_strategy, self.sensor_count, stable_seed, allow_boundary=True
        )
        measured = sample_field(central, indices, grid)
        mask = np.zeros(central.shape, dtype=bool)
        sparse = np.zeros(central.shape, dtype=float)
        mask[indices[:, 0], indices[:, 1]] = True
        sparse[indices[:, 0], indices[:, 1]] = measured
        return {
            "task_type": "external_heat_flux", "usage_role": "external_audit",
            "temperature_window": window, "central_temperature": central,
            "external_reference_flux": flux, "central_frame_index": central_frame_index,
            "window_frame_indices": selected_frames, "sensor_mask": mask,
            "sparse_temperature": sparse, "sensor_indices": indices.copy(),
            "measured_temperatures": measured.copy(), "experiment_id": self.manifest["experiment_id"],
            "plate_id": self.manifest["plate_id"], "time_semantics": "ordered_frame_index_only",
            "time_values": None, "temperature_units": "unresolved", "heat_flux_units": "unresolved",
            "target_provenance": self.manifest["target_provenance"],
            "independent_ground_truth": False, "classical_q_target": False,
            "sensor_configuration": {
                "strategy": self.sensor_strategy, "count": len(indices), "seed": self.sensor_seed,
                "provenance": self.manifest["sparse_sampling_provenance"],
            },
        }


def validate_synthetic_dataset(directory: str | Path) -> dict[str, Any]:
    with SyntheticDatasetReader(directory) as reader:
        for index in range(len(reader)):
            reader[index]
        return {"valid": True, "sample_count": len(reader), "task_type": "synthetic_source"}


def validate_external_manifest(path: str | Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetPipelineError("Missing or invalid external manifest") from exc
    if manifest.get("manifest_content_sha256") != _manifest_content_hash(manifest):
        raise DatasetPipelineError("External manifest checksum is invalid")
    required_values = {
        "task_type": "external_heat_flux",
        "usage_role": "external_audit",
        "target_provenance": "HFITS-derived external reference heat-flux estimate",
        "independent_ground_truth": False,
        "classical_q_target": False,
        "temperature_key_pattern": EXTERNAL_TEMPERATURE_PATTERN.pattern,
        "heat_flux_key_pattern": EXTERNAL_FLUX_PATTERN.pattern,
    }
    for key, expected in required_values.items():
        if manifest.get(key) != expected:
            raise DatasetPipelineError(f"External manifest {key} is invalid")
    files = []
    for file_key, size_key, checksum_key in (
        ("temperature_path", "temperature_size_bytes", "temperature_sha256"),
        ("heat_flux_path", "heat_flux_size_bytes", "heat_flux_sha256"),
    ):
        file_path = Path(manifest[file_key])
        if not file_path.is_file():
            raise DatasetPipelineError(f"External file is missing: {file_path}")
        if file_path.stat().st_size != manifest.get(size_key):
            raise DatasetPipelineError(f"External file size mismatch: {file_key}")
        recorded_checksum = manifest.get(checksum_key)
        if recorded_checksum is not None and _sha256_file(file_path) != recorded_checksum:
            raise DatasetPipelineError(f"External file checksum mismatch: {file_key}")
        files.append(file_path)
    try:
        with h5py.File(files[0], "r") as temperature, h5py.File(files[1], "r") as flux:
            temperature_frames = _external_frame_map(
                temperature, EXTERNAL_TEMPERATURE_PATTERN, "temperature"
            )
            flux_frames = _external_frame_map(flux, EXTERNAL_FLUX_PATTERN, "heat-flux")
            if set(temperature_frames) != set(flux_frames):
                raise DatasetPipelineError("Temperature and heat-flux frame-index sets differ")
            frames = sorted(temperature_frames)
            shapes = set()
            for frame in frames:
                temperature_shape = temperature[temperature_frames[frame]].shape
                flux_shape = flux[flux_frames[frame]].shape
                if temperature_shape != flux_shape:
                    raise DatasetPipelineError(f"External frame {frame} shapes differ")
                shapes.add(tuple(temperature_shape))
    except OSError as exc:
        raise DatasetPipelineError("External pair contains invalid HDF5") from exc
    if frames != manifest.get("frame_indices"):
        raise DatasetPipelineError("External frame indices differ from manifest")
    if len(frames) != manifest.get("frame_count"):
        raise DatasetPipelineError("External frame count differs from manifest")
    if len(shapes) != 1 or list(next(iter(shapes))) != manifest.get("frame_shape"):
        raise DatasetPipelineError("External frame shape differs from manifest")
    return {"valid": True, "frame_count": manifest["frame_count"], "usage_role": manifest["usage_role"]}


def preview_synthetic_dataset(directory: str | Path, output: str | Path, count: int) -> list[str]:
    """Create small truthful synthetic preview panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    created = []
    with SyntheticDatasetReader(directory) as reader:
        chosen = np.linspace(0, len(reader) - 1, min(count, len(reader)), dtype=int)
        for index in chosen:
            sample = reader[int(index)]
            fields = [
                ("true_source", "True source q (interior-node target)"),
                ("full_temperature", "Full temperature"),
                ("sensor_mask", "Sensor mask"), ("identity_reconstruction", "Identity Tikhonov"),
                ("smoothness_reconstruction", "Smoothness Tikhonov"),
            ]
            figure, axes = plt.subplots(1, len(fields), figsize=(15, 3), constrained_layout=True)
            for axis, (field, title) in zip(axes, fields):
                image = axis.imshow(sample[field], origin="lower")
                axis.set_title(title)
                axis.set_xlabel("grid j")
                axis.set_ylabel("grid i")
                figure.colorbar(image, ax=axis, shrink=0.75)
            figure.suptitle(f"{sample['sample_id']} | {sample['split']} | {sample['source_family']}")
            path = output_path / f"synthetic_{index:03d}.png"
            figure.savefig(path, dpi=120)
            plt.close(figure)
            created.append(str(path.resolve()))
    return created


def preview_external_window(
    reader: ExternalWindowReader, central_frame: int, output: str | Path
) -> list[str]:
    """Create a small external window panel and JSON metadata without copying raw arrays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    with reader:
        sample = reader.get(central_frame)
    fields = [
        (sample["temperature_window"][0], f"Window first frame {sample['window_frame_indices'][0]}"),
        (sample["central_temperature"], f"Central temperature frame {central_frame}"),
        (sample["sparse_temperature"], "Simulated sparse temperature"),
        (sample["external_reference_flux"], "HFITS-derived reference flux"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(13, 3), constrained_layout=True)
    for axis, (values, title) in zip(axes, fields):
        image = axis.imshow(values, origin="lower")
        axis.set_title(title + "\nunits unresolved")
        axis.set_xlabel("pixel column")
        axis.set_ylabel("pixel row")
        figure.colorbar(image, ax=axis, shrink=0.7)
    image_path = output_path / f"external_window_{central_frame:06d}.png"
    figure.savefig(image_path, dpi=120)
    plt.close(figure)
    metadata = {key: value for key, value in sample.items() if not isinstance(value, np.ndarray)}
    metadata_path = output_path / f"external_window_{central_frame:06d}.json"
    _atomic_json(metadata_path, metadata)
    return [str(image_path.resolve()), str(metadata_path.resolve())]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="inspect one local HDF5 file")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--sample-limit", type=int, default=4096)
    inspect_parser.add_argument("--json", action="store_true", help="emit JSON")
    smoke = subparsers.add_parser("build-smoke", help="build a tiny configured synthetic dataset")
    smoke.add_argument("config", type=Path)
    smoke.add_argument("--output", type=Path)
    build = subparsers.add_parser("build", help="build a configured synthetic dataset")
    build.add_argument("config", type=Path)
    build.add_argument("--output", type=Path)
    external = subparsers.add_parser("external-manifest", help="validate an external HDF5 pair")
    external.add_argument("temperature", type=Path)
    external.add_argument("heat_flux", type=Path)
    external.add_argument("output", type=Path)
    external.add_argument("--experiment-id", required=True)
    external.add_argument("--plate-id", required=True)
    external.add_argument("--skip-checksums", action="store_true")
    synthetic_preview = subparsers.add_parser("preview-synthetic", help="preview synthetic samples")
    synthetic_preview.add_argument("dataset_directory", type=Path)
    synthetic_preview.add_argument("output", type=Path)
    synthetic_preview.add_argument("--count", type=int, default=3)
    external_preview = subparsers.add_parser("preview-external", help="preview one external temporal window")
    external_preview.add_argument("manifest", type=Path)
    external_preview.add_argument("output", type=Path)
    external_preview.add_argument("--frame", type=int, required=True)
    external_preview.add_argument("--window-size", type=int, required=True)
    external_preview.add_argument("--boundary-policy", choices=("reject", "edge-repeat", "reflect"), required=True)
    external_preview.add_argument("--sensor-strategy", choices=("regular_grid", "random", "boundary", "center_focused"), required=True)
    external_preview.add_argument("--sensor-count", type=int, required=True)
    external_preview.add_argument("--sensor-seed", type=int, required=True)
    validate = subparsers.add_parser("validate", help="validate a generated dataset or external manifest")
    target = validate.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset-directory", type=Path)
    target.add_argument("--external-manifest", type=Path)
    validate_config_parser = subparsers.add_parser(
        "validate-config", help="validate a dataset configuration without generating data"
    )
    validate_config_parser.add_argument("config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            report = inspect_hdf5(args.path, sample_limit=args.sample_limit)
            print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_inspection(report))
        elif args.command in {"build-smoke", "build"}:
            report = build_synthetic_dataset(load_dataset_config(args.config), output_directory=args.output)
            print(json.dumps(report, indent=2, sort_keys=True))
        elif args.command == "external-manifest":
            report = build_external_manifest(
                args.temperature, args.heat_flux, args.output,
                experiment_id=args.experiment_id, plate_id=args.plate_id,
                calculate_checksums=not args.skip_checksums,
            )
            print(json.dumps({"manifest": str(args.output.resolve()), "frame_count": report["frame_count"]}, indent=2))
        elif args.command == "preview-synthetic":
            print(json.dumps(preview_synthetic_dataset(args.dataset_directory, args.output, args.count), indent=2))
        elif args.command == "preview-external":
            reader = ExternalWindowReader(
                args.manifest, window_size=args.window_size, boundary_policy=args.boundary_policy,
                sensor_strategy=args.sensor_strategy, sensor_count=args.sensor_count,
                sensor_seed=args.sensor_seed,
            )
            print(json.dumps(preview_external_window(reader, args.frame, args.output), indent=2))
        elif args.command == "validate-config":
            config = load_dataset_config(args.config)
            plans = plan_synthetic_samples(config)
            print(json.dumps({
                "valid": True, "schema_version": config["schema_version"],
                "planned_samples": len(plans),
                "sensor_strategies": sorted({plan["sensor_strategy"] for plan in plans}),
            }, indent=2, sort_keys=True))
        else:
            report = (
                validate_synthetic_dataset(args.dataset_directory)
                if args.dataset_directory else validate_external_manifest(args.external_manifest)
            )
            print(json.dumps(report, indent=2, sort_keys=True))
    except (FileNotFoundError, HDF5InspectionError, DatasetPipelineError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
