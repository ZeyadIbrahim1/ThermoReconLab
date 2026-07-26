"""Task 4 synthetic benchmark evaluation and uncertainty analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_data import DatasetPipelineError, SyntheticDatasetReader, validate_synthetic_dataset  # noqa: E402
from ai_model import (  # noqa: E402
    SyntheticTorchDataset, build_model, denormalize_source, load_checkpoint,
    parameter_count, train, validate_model_config,
)
from thermoreconlab.core.grid import Grid2D  # noqa: E402
from thermoreconlab.reconstruction import solve_forward  # noqa: E402


EVALUATION_SCHEMA_VERSION = 1
TEST_ROLES = ("test_id", "test_ood_shape", "test_ood_sensor", "test_ood_noise")
METHODS = ("full_residual_attention", "residual_no_attention", "direct_sparse_mask", "identity", "smoothness")
TRUTH_LABEL = "Synthetic benchmark only\nNo external generalization claim"
RUN_DEFINITIONS = {
    "full_residual_attention": (True, "residual", [1, 1, 1, 1]),
    "residual_no_attention": (False, "residual", [1, 1, 1, 1]),
    "direct_sparse_mask": (True, "direct", [1, 1, 0, 0]),
}
IDENTITY_FIELDS = {"sample_id", "method", "test_role", "source_family", "sensor_strategy", "noise_level", "sensor_count_bin"}
STATISTIC_FIELDS = {
    "source_squared_error_sum", "source_absolute_error_sum", "source_target_squared_sum",
    "source_valid_node_count", "source_global_maximum_absolute", "source_integral_error_sum",
    "nonnegative_violation_count", "boundary_value_count", "boundary_global_maximum_absolute",
    "temperature_squared_error_sum", "temperature_absolute_error_sum", "temperature_node_count",
    "clean_sensor_squared_error_sum", "clean_sensor_count", "clean_sensor_reference_squared_sum",
    "noisy_sensor_squared_error_sum", "noisy_sensor_count", "noisy_sensor_reference_squared_sum",
}


class EvaluationError(RuntimeError):
    """Raised when Task 4 configuration, leakage, or evaluation is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read evaluation configuration: {path}") from exc
    value["_configuration_path"] = str(Path(path).resolve())
    return validate_evaluation_config(value)


def validate_evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "seed", "partition_seed", "dataset_directory",
        "base_model_configuration", "validation_select_count",
        "validation_calibration_count", "model_runs", "checkpoint_directory",
        "output_directory", "log_directory", "test_roles", "bootstrap_repetitions",
        "bootstrap_confidence_level", "uncertainty_method", "mc_dropout_passes",
        "uncertainty_std_floor", "target_interval_coverage", "evaluation_batch_size",
        "preview_sample_count", "physics_evaluation_enabled", "external_manifest",
        "sensor_count_bins",
    }
    if not isinstance(config, dict) or required - config.keys():
        raise EvaluationError(f"Missing evaluation keys: {sorted(required - config.keys())}")
    if config["schema_version"] != EVALUATION_SCHEMA_VERSION:
        raise EvaluationError("Unsupported evaluation schema")
    for key in ("seed", "partition_seed", "validation_select_count", "validation_calibration_count", "bootstrap_repetitions", "mc_dropout_passes", "evaluation_batch_size", "preview_sample_count"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EvaluationError(f"{key} must be a positive integer")
    if set(config["test_roles"]) != set(TEST_ROLES) or len(config["test_roles"]) != 4:
        raise EvaluationError("test_roles must contain every Task 4 test role exactly once")
    if config["uncertainty_method"] != "mc_dropout":
        raise EvaluationError("Only mc_dropout uncertainty is supported")
    for key in ("bootstrap_confidence_level", "target_interval_coverage"):
        if not 0 < float(config[key]) < 1:
            raise EvaluationError(f"{key} must lie in (0, 1)")
    if float(config["uncertainty_std_floor"]) <= 0:
        raise EvaluationError("uncertainty_std_floor must be positive")
    if config["physics_evaluation_enabled"] is not True:
        raise EvaluationError("physics_evaluation_enabled must be true for the Task 4 benchmark")
    if config["external_manifest"] is not None and not isinstance(config["external_manifest"], (str, Path)):
        raise EvaluationError("external_manifest must be null or a path")
    expected_runs = set(RUN_DEFINITIONS)
    runs = config["model_runs"]
    if not isinstance(runs, list) or len(runs) != 3 or any(not isinstance(run, dict) for run in runs) or {run.get("name") for run in runs} != expected_runs:
        raise EvaluationError("Exactly the three fixed Task 4 runs are required")
    required_run = {"name", "attention", "prediction_mode", "input_channel_mask", "epochs", "batch_size", "workers"}
    for run in runs:
        missing = required_run - run.keys()
        if missing:
            raise EvaluationError(f"Incomplete model run {run.get('name')!r}: {sorted(missing)}")
        if set(run) - (required_run | {"device_policy"}):
            raise EvaluationError(f"Unknown fields in model run {run['name']}")
        expected = RUN_DEFINITIONS[run["name"]]
        actual = (run["attention"], run["prediction_mode"], run["input_channel_mask"])
        if actual != expected:
            raise EvaluationError(f"Invalid fixed architecture declaration for {run['name']}")
        for key in ("epochs", "batch_size"):
            if isinstance(run[key], bool) or not isinstance(run[key], int) or run[key] < 1:
                raise EvaluationError(f"model_runs.{run['name']}.{key} must be positive")
        if isinstance(run["workers"], bool) or not isinstance(run["workers"], int) or run["workers"] < 0:
            raise EvaluationError(f"model_runs.{run['name']}.workers must be nonnegative")
        if "device_policy" in run and run["device_policy"] not in {"auto", "cpu", "cuda"}:
            raise EvaluationError(f"Invalid device_policy for {run['name']}")
    bins = config["sensor_count_bins"]
    if (not isinstance(bins, list) or len(bins) < 2 or
            any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in bins) or
            any(a >= b for a, b in zip(bins, bins[1:]))):
        raise EvaluationError("sensor_count_bins must be strictly increasing")
    manifest_path = Path(config["dataset_directory"]) / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            total_test = sum(1 for sample in manifest["samples"] if sample["split"] in TEST_ROLES)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EvaluationError("Cannot inspect dataset manifest for preview validation") from exc
        if config["preview_sample_count"] > total_test:
            raise EvaluationError("preview_sample_count exceeds total test samples")
    return config


def partition_validation(
    dataset_directory: str | Path, *, partition_seed: int,
    select_count: int, calibration_count: int, output_path: str | Path | None = None,
) -> dict[str, Any]:
    reader = SyntheticDatasetReader(dataset_directory)
    try:
        train_ids = [s["sample_id"] for s in reader.manifest["samples"] if s["split"] == "train"]
        validation = [s["sample_id"] for s in reader.manifest["samples"] if s["split"] == "validation"]
    finally:
        reader.close()
    if select_count + calibration_count != len(validation):
        raise EvaluationError("Validation partition counts must cover validation exactly")
    ranked = sorted(validation, key=lambda sample_id: hashlib.sha256(f"{partition_seed}:{sample_id}".encode()).hexdigest())
    select_ids = ranked[:select_count]
    calibration_ids = ranked[select_count:]
    manifest: dict[str, Any] = {
        "schema_version": 1, "partition_seed": partition_seed,
        "reason": "stable_sha256(partition_seed + sample_id)",
        "train_sample_ids": train_ids,
        "validation_select_sample_ids": select_ids,
        "validation_calibration_sample_ids": calibration_ids,
        "validation_source_count": len(validation),
    }
    manifest["partition_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    if output_path is not None:
        _atomic_json(Path(output_path), manifest)
    return manifest


def dataset_hashes_from_manifest(dataset_directory: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
    path = Path(dataset_directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        hashes = {
            "dataset_manifest_hash": manifest["manifest_content_sha256"],
            "dataset_hdf5_hash": manifest["dataset_sha256"],
            "configuration_hash": manifest["configuration_sha256"],
            "normalization_hash": manifest["normalization_sha256"],
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EvaluationError("Cannot read synthetic dataset manifest hashes") from exc
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise EvaluationError("Synthetic dataset manifest contains invalid hashes")
    return hashes, manifest


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def verify_checkpoints(
    config: dict[str, Any], partition: dict[str, Any], run_configs: dict[str, dict[str, Any]],
    *, output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify all checkpoint provenance before exposing any model for inference."""
    expected_dataset, manifest = dataset_hashes_from_manifest(config["dataset_directory"])
    by_role = defaultdict(list)
    for sample in manifest["samples"]:
        by_role[sample["split"]].append(sample["sample_id"])
    train_ids = partition.get("train_sample_ids"); validation_ids = partition.get("validation_select_sample_ids")
    calibration_ids = partition.get("validation_calibration_sample_ids")
    if any(not isinstance(ids, list) or len(ids) != len(set(ids)) for ids in (train_ids, validation_ids, calibration_ids)):
        raise EvaluationError("Partition IDs must be duplicate-free lists")
    if train_ids != by_role["train"]:
        raise EvaluationError("Partition train IDs do not exactly match the dataset train split")
    expected_validation = partition_validation(
        config["dataset_directory"], partition_seed=config["partition_seed"],
        select_count=config["validation_select_count"], calibration_count=config["validation_calibration_count"],
    )
    if validation_ids != expected_validation["validation_select_sample_ids"] or calibration_ids != expected_validation["validation_calibration_sample_ids"]:
        raise EvaluationError("Partition validation-selection/calibration IDs are not the expected deterministic split")
    test_ids = {sample_id for role in TEST_ROLES for sample_id in by_role[role]}
    shared_train = shared_validation = None
    records: dict[str, Any] = {}
    for name in RUN_DEFINITIONS:
        run_config = run_configs[name]
        checkpoint_path = Path(config["checkpoint_directory"]) / name / "best.pt"
        try:
            checkpoint = load_checkpoint(checkpoint_path, expected_dataset=expected_dataset)
        except Exception as exc:
            raise EvaluationError(f"Checkpoint verification failed for {name}: {exc}") from exc
        checkpoint_train = checkpoint.get("train_sample_ids"); checkpoint_validation = checkpoint.get("validation_sample_ids")
        if not isinstance(checkpoint_train, list) or len(checkpoint_train) != len(set(checkpoint_train)):
            raise EvaluationError(f"Checkpoint {name} train IDs contain duplicates or are invalid")
        if not isinstance(checkpoint_validation, list) or len(checkpoint_validation) != len(set(checkpoint_validation)):
            raise EvaluationError(f"Checkpoint {name} validation IDs contain duplicates or are invalid")
        if checkpoint_train != train_ids:
            raise EvaluationError(f"Checkpoint {name} train IDs do not match the partition")
        if checkpoint_validation != validation_ids:
            raise EvaluationError(f"Checkpoint {name} validation-selection IDs do not match the partition")
        if set(checkpoint_validation) & set(calibration_ids):
            raise EvaluationError(f"Checkpoint {name} contains calibration leakage")
        if (set(checkpoint_train) | set(checkpoint_validation)) & test_ids:
            raise EvaluationError(f"Checkpoint {name} contains test leakage")
        if checkpoint.get("epoch") != checkpoint.get("best_epoch"):
            raise EvaluationError(f"Checkpoint {name} best.pt epoch differs from best_epoch")
        if checkpoint.get("model_architecture") != run_config.get("architecture"):
            raise EvaluationError(f"Checkpoint {name} architecture differs from declared run architecture")
        if checkpoint.get("nonnegative_policy") != run_config.get("nonnegative_policy"):
            raise EvaluationError(f"Checkpoint {name} nonnegative policy differs from the run configuration")
        training_config = checkpoint.get("training_configuration", {})
        if not isinstance(training_config, dict) or not _same_path(training_config.get("dataset_directory", ""), config["dataset_directory"]):
            raise EvaluationError(f"Checkpoint {name} training configuration points to another dataset")
        if shared_train is None:
            shared_train, shared_validation = checkpoint_train, checkpoint_validation
        elif checkpoint_train != shared_train or checkpoint_validation != shared_validation:
            raise EvaluationError("The three checkpoints do not share train and validation-selection IDs")
        records[name] = {
            "verified": True, "checkpoint": str(checkpoint_path), "epoch": int(checkpoint["epoch"]),
            "best_epoch": int(checkpoint["best_epoch"]), "train_sample_count": len(checkpoint_train),
            "validation_selection_sample_count": len(checkpoint_validation),
            "architecture": checkpoint["model_architecture"], "nonnegative_policy": checkpoint["nonnegative_policy"],
        }
    report = {
        "verified": True, "dataset_hashes": expected_dataset,
        "partition": {"train_sample_count": len(train_ids), "validation_selection_sample_count": len(validation_ids), "validation_calibration_sample_count": len(calibration_ids), "test_sample_count": len(test_ids)},
        "checks": ["dataset hashes", "duplicate-free IDs", "exact train IDs", "exact validation-selection IDs", "no calibration leakage", "no test leakage", "best epoch", "architecture", "nonnegative policy", "training dataset path", "shared model partitions"],
        "models": records,
    }
    if output_path is not None:
        _atomic_json(Path(output_path), report)
    return report


class EvaluationDataset:
    """Lazy physical-array adapter for validation or fixed test roles."""

    def __init__(self, directory: str | Path, role: str, sample_ids: list[str] | None = None):
        allowed = {"validation", *TEST_ROLES}
        if role not in allowed:
            raise EvaluationError(f"Evaluation role is not allowed: {role}")
        self.reader = SyntheticDatasetReader(directory)
        available = {s["sample_id"]: i for i, s in enumerate(self.reader.manifest["samples"]) if s["split"] == role}
        ids = list(available) if sample_ids is None else list(sample_ids)
        if len(ids) != len(set(ids)):
            self.reader.close(); raise EvaluationError("Duplicate evaluation sample ID")
        unknown = sorted(set(ids) - set(available))
        if unknown:
            self.reader.close(); raise EvaluationError(f"Unknown or wrong-role sample IDs: {unknown}")
        self.indices = [available[sample_id] for sample_id in ids]
        self.sample_ids = ids
        self.normalization = self.reader.normalization

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.reader[self.indices[index]]

    def close(self) -> None:
        self.reader.close()


def source_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, float); target = np.asarray(target, float); mask = np.asarray(mask, bool)
    if prediction.shape != target.shape or mask.shape != target.shape or not mask.any():
        raise EvaluationError("Source metric shapes are incompatible or mask is empty")
    error = prediction[mask] - target[mask]
    squared_sum = float(np.sum(error ** 2)); absolute_sum = float(np.sum(np.abs(error)))
    target_squared_sum = float(np.sum(target[mask] ** 2)); valid_count = int(mask.sum())
    pred_peak = np.unravel_index(np.argmax(np.where(mask, prediction, -np.inf)), prediction.shape)
    target_peak = np.unravel_index(np.argmax(np.where(mask, target, -np.inf)), target.shape)
    boundary = np.ones(mask.shape, bool); boundary[1:-1, 1:-1] = False
    boundary_count = int(boundary.sum()); maximum = float(np.max(np.abs(error)))
    integral_error = float(np.sum(prediction[mask]) - np.sum(target[mask]))
    nonnegative_count = int(np.count_nonzero(prediction[mask] < 0))
    boundary_maximum = float(np.max(np.abs(prediction[boundary])))
    return {
        "rmse": float(np.sqrt(squared_sum / valid_count)),
        "mae": absolute_sum / valid_count,
        "relative_l2": float(np.sqrt(squared_sum / max(target_squared_sum, np.finfo(float).eps))),
        "maximum_absolute_error": maximum,
        "source_integral_error": integral_error,
        "source_peak_value_error": float(prediction[pred_peak] - target[target_peak]),
        "source_peak_location_distance": float(np.linalg.norm(np.subtract(pred_peak, target_peak))),
        "nonnegative_violation_fraction": nonnegative_count / valid_count,
        "boundary_maximum_absolute": boundary_maximum,
        "source_squared_error_sum": squared_sum,
        "source_absolute_error_sum": absolute_sum,
        "source_target_squared_sum": target_squared_sum,
        "source_valid_node_count": valid_count,
        "source_global_maximum_absolute": maximum,
        "source_integral_error_sum": integral_error,
        "nonnegative_violation_count": nonnegative_count,
        "boundary_value_count": boundary_count,
        "boundary_global_maximum_absolute": boundary_maximum,
    }


def physics_metrics(prediction: np.ndarray, sample: dict[str, Any]) -> dict[str, float]:
    mask = np.asarray(sample["source_valid_mask"], bool)
    source = np.asarray(prediction, float) * mask
    grid = Grid2D(*tuple(sample["grid_shape"]))
    predicted_temperature = solve_forward(source, grid)
    full = np.asarray(sample["full_temperature"], float)
    indices = np.asarray(sample["sensor_indices"], int)
    clean = full[indices[:, 0], indices[:, 1]]
    noisy = np.asarray(sample["measured_temperatures"], float)
    predicted_sensors = predicted_temperature[indices[:, 0], indices[:, 1]]
    full_error = predicted_temperature - full
    clean_error = predicted_sensors - clean
    noisy_error = predicted_sensors - noisy
    temperature_squared = float(np.sum(full_error ** 2)); temperature_absolute = float(np.sum(np.abs(full_error)))
    clean_squared = float(np.sum(clean_error ** 2)); noisy_squared = float(np.sum(noisy_error ** 2))
    clean_reference = float(np.sum(clean ** 2)); noisy_reference = float(np.sum(noisy ** 2))
    return {
        "physics_temperature_rmse": float(np.sqrt(np.mean(full_error ** 2))),
        "physics_temperature_mae": float(np.mean(np.abs(full_error))),
        "clean_sensor_residual_rms": float(np.sqrt(np.mean(clean_error ** 2))),
        "noisy_measurement_residual_rms": float(np.sqrt(np.mean(noisy_error ** 2))),
        "clean_relative_sensor_residual": float(np.linalg.norm(clean_error) / max(np.linalg.norm(clean), np.finfo(float).eps)),
        "noisy_relative_sensor_residual": float(np.linalg.norm(noisy_error) / max(np.linalg.norm(noisy), np.finfo(float).eps)),
        "temperature_squared_error_sum": temperature_squared,
        "temperature_absolute_error_sum": temperature_absolute,
        "temperature_node_count": int(full.size),
        "clean_sensor_squared_error_sum": clean_squared,
        "clean_sensor_count": int(clean.size),
        "clean_sensor_reference_squared_sum": clean_reference,
        "noisy_sensor_squared_error_sum": noisy_squared,
        "noisy_sensor_count": int(noisy.size),
        "noisy_sensor_reference_squared_sum": noisy_reference,
    }


def _model_input(sample: dict[str, Any], normalization: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stats = normalization["statistics"]
    norm = lambda x, key: (np.asarray(x, np.float32) - float(stats[key]["mean"])) / float(stats[key]["scale"])
    sparse = norm(sample["sparse_temperature"], "sparse_temperature")
    identity = norm(sample["identity_reconstruction"], "true_source")
    smooth = norm(sample["smoothness_reconstruction"], "true_source")
    inputs = np.stack((sparse, np.asarray(sample["sensor_mask"], np.float32), identity, smooth))[None]
    mask = np.asarray(sample["source_valid_mask"], np.float32)[None, None]
    return torch.from_numpy(inputs).to(device), torch.from_numpy(smooth[None, None]).to(device), torch.from_numpy(mask).to(device)


def load_trained_model(config: dict[str, Any], checkpoint_path: str | Path, normalization: dict[str, Any], device: torch.device, *, expected_dataset: dict[str, str] | None = None) -> nn.Module:
    model = build_model(config, normalization).to(device)
    checkpoint = load_checkpoint(checkpoint_path, expected_dataset=expected_dataset)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_sample(model: nn.Module, sample: dict[str, Any], normalization: dict[str, Any], device: torch.device) -> np.ndarray:
    inputs, smooth, mask = _model_input(sample, normalization, device)
    with torch.no_grad():
        normalized = model(inputs, smooth, mask)
        physical = denormalize_source(normalized, normalization, mask)
    return physical[0, 0].cpu().numpy()


def predict_batch(model: nn.Module, samples: list[dict[str, Any]], normalization: dict[str, Any], device: torch.device) -> list[np.ndarray]:
    """Predict a bounded in-memory batch; the backing HDF5 reader remains lazy."""
    if not samples:
        return []
    prepared = [_model_input(sample, normalization, device) for sample in samples]
    inputs = torch.cat([item[0] for item in prepared]); smooth = torch.cat([item[1] for item in prepared]); mask = torch.cat([item[2] for item in prepared])
    with torch.no_grad():
        physical = denormalize_source(model(inputs, smooth, mask), normalization, mask)
    return [array for array in physical[:, 0].cpu().numpy()]


def aggregate_metrics(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    numeric = [key for key in rows[0] if key not in IDENTITY_FIELDS | STATISTIC_FIELDS and isinstance(rows[0][key], (int, float))] if rows else []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        identity = dict(zip(group_keys, group)); mean_record = dict(identity, aggregation_type="mean_per_sample", sample_count=len(values))
        for key in numeric:
            present = [float(value[key]) for value in values if key in value]
            if present: mean_record[key] = float(np.mean(present))
        output.append(mean_record)
        pooled = dict(identity, aggregation_type="pooled_global", sample_count=len(values))
        for key in STATISTIC_FIELDS:
            present = [float(value[key]) for value in values if key in value]
            if present: pooled[key] = float(sum(present))
        if "source_valid_node_count" in pooled:
            pooled["source_global_maximum_absolute"] = max(float(value["source_global_maximum_absolute"]) for value in values)
            pooled["boundary_global_maximum_absolute"] = max(float(value["boundary_global_maximum_absolute"]) for value in values)
            pooled["rmse"] = math.sqrt(pooled["source_squared_error_sum"] / pooled["source_valid_node_count"])
            pooled["mae"] = pooled["source_absolute_error_sum"] / pooled["source_valid_node_count"]
            pooled["relative_l2"] = math.sqrt(pooled["source_squared_error_sum"] / max(pooled["source_target_squared_sum"], np.finfo(float).eps))
            pooled["maximum_absolute_error"] = pooled["source_global_maximum_absolute"]
            pooled["source_integral_error"] = pooled["source_integral_error_sum"]
            pooled["nonnegative_violation_fraction"] = pooled["nonnegative_violation_count"] / pooled["source_valid_node_count"]
            pooled["boundary_maximum_absolute"] = pooled["boundary_global_maximum_absolute"]
        if "temperature_node_count" in pooled:
            pooled["physics_temperature_rmse"] = math.sqrt(pooled["temperature_squared_error_sum"] / pooled["temperature_node_count"])
            pooled["physics_temperature_mae"] = pooled["temperature_absolute_error_sum"] / pooled["temperature_node_count"]
            for prefix, rms_name, relative_name in (("clean", "clean_sensor_residual_rms", "clean_relative_sensor_residual"), ("noisy", "noisy_measurement_residual_rms", "noisy_relative_sensor_residual")):
                pooled[rms_name] = math.sqrt(pooled[f"{prefix}_sensor_squared_error_sum"] / pooled[f"{prefix}_sensor_count"])
                pooled[relative_name] = math.sqrt(pooled[f"{prefix}_sensor_squared_error_sum"] / max(pooled[f"{prefix}_sensor_reference_squared_sum"], np.finfo(float).eps))
        output.append(pooled)
    return output


def paired_bootstrap(
    learned: np.ndarray, baseline: np.ndarray, *, repetitions: int,
    confidence: float, seed: int,
) -> dict[str, Any]:
    learned = np.asarray(learned, float); baseline = np.asarray(baseline, float)
    if learned.shape != baseline.shape or learned.ndim != 1 or not len(learned):
        raise EvaluationError("Paired bootstrap requires equal non-empty vectors")
    differences = learned - baseline
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for index in range(repetitions): means[index] = np.mean(differences[rng.integers(0, len(differences), len(differences))])
    alpha = (1 - confidence) / 2
    return {
        "difference_convention": "learned metric - baseline metric",
        "mean_paired_difference": float(np.mean(differences)),
        "median_paired_difference": float(np.median(differences)),
        "confidence_interval": [float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))],
        "win_rate": float(np.mean(differences < 0)), "sample_count": len(differences),
    }


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def mc_dropout_prediction(
    model: nn.Module, sample: dict[str, Any], normalization: dict[str, Any],
    device: torch.device, *, passes: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    enable_mc_dropout(model)
    predictions = []
    with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
        torch.manual_seed(seed)
        if device.type == "cuda": torch.cuda.manual_seed_all(seed)
        for _ in range(passes): predictions.append(predict_sample(model, sample, normalization, device))
    values = np.stack(predictions)
    model.eval()
    return values.mean(0), values.std(0, ddof=0)


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, float).ravel(); right = np.asarray(right, float).ravel()
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise EvaluationError("Spearman vectors must have equal shapes and finite values")
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable"); ranks = np.empty(len(values), float)
        sorted_values = values[order]; start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and sorted_values[stop] == sorted_values[start]: stop += 1
            ranks[order[start:stop]] = (start + stop - 1) / 2.0
            start = stop
        return ranks
    return float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1])


def conformal_multiplier(scores: np.ndarray, coverage: float) -> float:
    scores = np.sort(np.asarray(scores, float).ravel())
    if not len(scores) or not np.isfinite(scores).all(): raise EvaluationError("Calibration scores must be finite and non-empty")
    rank = min(int(math.ceil((len(scores) + 1) * coverage)), len(scores))
    return float(scores[rank - 1])


def external_compatibility(manifest_path: str | Path | None) -> dict[str, Any]:
    if manifest_path is None:
        return {"decision": "no-go", "configured": False, "reason": "No external manifest configured"}
    try: manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise EvaluationError("Cannot read external metadata manifest") from exc
    compatible_metadata = manifest.get("task_type") == "external_heat_flux" and manifest.get("classical_q_target") is False
    return {
        "decision": "no-go", "configured": True, "metadata_conditions_verified": compatible_metadata,
        "external_task_type": manifest.get("task_type"), "model_task_type": "synthetic_source",
        "classical_q_target": manifest.get("classical_q_target"),
        "reasons": ["target mismatch", "unresolved physical units", "unresolved coordinates", "transient versus steady-state physics mismatch", "no valid classical q interpretation", "no experiment-level external generalization split"],
        "arrays_opened": False, "inference_refused": True,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def select_preview_samples(dataset_directory: str | Path, roles: Iterable[str], count: int, seed: int) -> list[tuple[str, str]]:
    """Round-robin deterministic samples across roles, with stable hashed ordering."""
    _, manifest = dataset_hashes_from_manifest(dataset_directory)
    ordered: dict[str, list[str]] = {}
    role_list = list(roles)
    for role in role_list:
        ids = [sample["sample_id"] for sample in manifest["samples"] if sample["split"] == role]
        ordered[role] = sorted(ids, key=lambda sample_id: hashlib.sha256(f"{seed}:{role}:{sample_id}".encode()).hexdigest())
    selected: list[tuple[str, str]] = []; index = 0
    while len(selected) < count:
        progressed = False
        for role in role_list:
            if index < len(ordered[role]) and len(selected) < count:
                selected.append((role, ordered[role][index])); progressed = True
        if not progressed: break
        index += 1
    if len(selected) != count:
        raise EvaluationError("preview_sample_count exceeds available test samples")
    return selected


def _compose_run_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_path = Path(config["base_model_configuration"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    logs = Path(config["log_directory"]); selection_path = logs / "validation_partition.json"
    run_configs = {}
    for run in config["model_runs"]:
        name = run["name"]; run_config = json.loads(json.dumps(base))
        run_config["run_label"] = f"Task 4 {name} - Synthetic benchmark only"
        run_config["seed"] = config["seed"]
        run_config["dataset_directory"] = config["dataset_directory"]
        run_config["sample_selection_manifest"] = str(selection_path)
        run_config["checkpoint_directory"] = str(Path(config["checkpoint_directory"]) / name)
        run_config["log_directory"] = str(logs / name)
        run_config["output_directory"] = str(Path(config["output_directory"]) / "training" / name)
        run_config["epochs"] = int(run["epochs"])
        run_config["batch_size"] = int(run["batch_size"])
        run_config["workers"] = int(run["workers"])
        run_config["device_policy"] = run.get("device_policy", run_config["device_policy"])
        run_config["architecture"].update({key: run[key] for key in ("attention", "prediction_mode", "input_channel_mask")})
        validate_model_config(run_config)
        run_configs[name] = run_config
    return run_configs


def _run_model_configs(config: dict[str, Any], partition: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    del partition
    logs = Path(config["log_directory"]); summaries = {}; run_configs = _compose_run_configs(config)
    for name, run_config in run_configs.items():
        summaries[name] = train(run_config)
    _atomic_json(logs / "training_runs.json", summaries)
    return summaries, run_configs


def _evaluate(config: dict[str, Any], run_configs: dict[str, dict[str, Any]], expected_dataset: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = SyntheticDatasetReader(config["dataset_directory"]); normalization = probe.normalization; probe.close()
    models = {name: load_trained_model(rc, Path(config["checkpoint_directory"]) / name / "best.pt", normalization, device, expected_dataset=expected_dataset) for name, rc in run_configs.items()}
    rows = []; test_manifest = {}
    bins = config["sensor_count_bins"]
    for role in config["test_roles"]:
        dataset = EvaluationDataset(config["dataset_directory"], role); test_manifest[role] = dataset.sample_ids
        batch_size = config["evaluation_batch_size"]
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            learned = {name: predict_batch(model, samples, normalization, device) for name, model in models.items()}
            for offset, sample in enumerate(samples):
                predictions = {name: values[offset] for name, values in learned.items()}
                predictions.update(identity=sample["identity_reconstruction"], smoothness=sample["smoothness_reconstruction"])
                count = int(sample["sensor_count"]); bin_index = max(0, min(np.searchsorted(bins, count, side="right") - 1, len(bins) - 2))
                for name, prediction in predictions.items():
                    row = {"sample_id": sample["sample_id"], "method": name, "test_role": role, "source_family": sample["source_family"], "sensor_strategy": sample["sensor_configuration"]["strategy"], "noise_level": float(sample["noise_configuration"]["relative_std"]), "sensor_count_bin": f"[{bins[bin_index]},{bins[bin_index + 1]})"}
                    row.update(source_metrics(prediction, sample["true_source"], sample["source_valid_mask"]))
                    if config["physics_evaluation_enabled"]: row.update(physics_metrics(prediction, sample))
                    rows.append(row)
        dataset.close()
    _atomic_json(Path(config["log_directory"]) / "test_sample_ids.json", test_manifest)
    aggregate = aggregate_metrics(rows, ("method", "test_role"))
    for record in aggregate_metrics(rows, ("method",)):
        record["test_role"] = "all_test_roles"
        aggregate.append(record)
    return rows, aggregate


def _bootstrap_outputs(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(r["sample_id"], r["method"]): r for r in rows}; output = {}
    for role in (*TEST_ROLES, "all_test_roles"):
        ids = sorted({r["sample_id"] for r in rows if role == "all_test_roles" or r["test_role"] == role})
        for learned in METHODS[:3]:
            for baseline in ("smoothness", "identity"):
                for metric in ("rmse", "mae", "relative_l2", "physics_temperature_rmse"):
                    key = f"{role}/{learned}/{baseline}/{metric}"
                    output[key] = paired_bootstrap(np.array([lookup[(i, learned)][metric] for i in ids]), np.array([lookup[(i, baseline)][metric] for i in ids]), repetitions=config["bootstrap_repetitions"], confidence=config["bootstrap_confidence_level"], seed=config["seed"])
    return output


def _uncertainty(config: dict[str, Any], partition: dict[str, Any], primary_config: dict[str, Any], expected_dataset: dict[str, str] | None = None) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = SyntheticDatasetReader(config["dataset_directory"]); normalization = probe.normalization; probe.close()
    model = load_trained_model(primary_config, Path(config["checkpoint_directory"]) / "full_residual_attention" / "best.pt", normalization, device, expected_dataset=expected_dataset)
    floor = float(config["uncertainty_std_floor"]); calibration = EvaluationDataset(config["dataset_directory"], "validation", partition["validation_calibration_sample_ids"])
    scores = []; calibration_pixels = 0
    for index in range(len(calibration)):
        sample = calibration[index]; mean, std = mc_dropout_prediction(model, sample, normalization, device, passes=config["mc_dropout_passes"], seed=config["seed"] + index)
        mask = sample["source_valid_mask"].astype(bool); scores.extend((np.abs(mean - sample["true_source"])[mask] / np.maximum(std[mask], floor)).tolist()); calibration_pixels += int(mask.sum())
    calibration.close(); multiplier = conformal_multiplier(np.asarray(scores), config["target_interval_coverage"])
    role_results = {}; all_uncertainty = []; all_error = []; per_sample = []; undefined_count = 0; global_index = 0
    for role in config["test_roles"]:
        dataset = EvaluationDataset(config["dataset_directory"], role); covered = total = 0; widths = []; sample_coverages = []
        role_uncertainty = []; role_error = []; role_correlations = []
        for index in range(len(dataset)):
            sample = dataset[index]; mean, std = mc_dropout_prediction(model, sample, normalization, device, passes=config["mc_dropout_passes"], seed=config["seed"] + 10000 + global_index); global_index += 1
            mask = sample["source_valid_mask"].astype(bool); radius = multiplier * np.maximum(std, floor); lower = np.maximum(mean - radius, 0.0); upper = mean + radius; truth = sample["true_source"]
            absolute_error = np.abs(mean - truth); valid_std = std[mask]; valid_error = absolute_error[mask]; valid_width = (upper - lower)[mask]
            inside = (truth >= lower) & (truth <= upper) & mask; sample_coverage = float(inside.sum() / mask.sum())
            correlation = spearman_correlation(valid_std, valid_error)
            if correlation is None: undefined_count += 1
            else: role_correlations.append(correlation)
            covered += int(inside.sum()); total += int(mask.sum()); sample_coverages.append(sample_coverage); widths.extend(valid_width.tolist())
            role_uncertainty.extend(valid_std.tolist()); role_error.extend(valid_error.tolist()); all_uncertainty.extend(valid_std.tolist()); all_error.extend(valid_error.tolist())
            per_sample.append({
                "sample_id": sample["sample_id"], "test_role": role,
                "mean_predictive_uncertainty": float(np.mean(valid_std)), "maximum_predictive_uncertainty": float(np.max(valid_std)),
                "mean_absolute_error": float(np.mean(valid_error)), "maximum_absolute_error": float(np.max(valid_error)),
                "uncertainty_error_spearman": correlation, "pixel_coverage": sample_coverage,
                "mean_interval_width": float(np.mean(valid_width)), "median_interval_width": float(np.median(valid_width)),
            })
        dataset.close(); role_results[role] = {
            "pixel_coverage": covered / total,
            "mean_per_sample_pixel_coverage": float(np.mean(sample_coverages)),
            "mean_interval_width": float(np.mean(widths)), "median_interval_width": float(np.median(widths)),
            "sample_count": len(sample_coverages),
            "pooled_pixel_uncertainty_error_spearman": spearman_correlation(np.asarray(role_uncertainty), np.asarray(role_error)),
            "mean_per_sample_uncertainty_error_spearman": float(np.mean(role_correlations)) if role_correlations else None,
            "undefined_constant_vector_correlation_count": len(sample_coverages) - len(role_correlations),
        }
    defined = [row["uncertainty_error_spearman"] for row in per_sample if row["uncertainty_error_spearman"] is not None]
    return {
        "method": "MC-dropout predictive dispersion", "target_coverage": config["target_interval_coverage"],
        "multiplier": multiplier, "std_floor": floor,
        "standard_deviation_floor_policy": "max(predictive_std, std_floor) is used for calibration scores and interval radii",
        "lower_bound_clipping_policy": "lower bounds are clipped to physical source value 0; upper bounds are not clipped",
        "coverage_note": "mean_per_sample_pixel_coverage is the arithmetic mean of sample interior-pixel coverages and equals pooled pixel coverage when all samples have equal valid interior-pixel counts",
        "calibration_sample_count": len(partition["validation_calibration_sample_ids"]), "calibration_pixel_count": calibration_pixels,
        "calibration_sample_ids": partition["validation_calibration_sample_ids"], "test_sample_count": len(per_sample),
        "test_role_results": role_results,
        "pooled_pixel_uncertainty_error_spearman": spearman_correlation(np.asarray(all_uncertainty), np.asarray(all_error)),
        "mean_per_sample_uncertainty_error_spearman": float(np.mean(defined)) if defined else None,
        "undefined_constant_vector_correlation_count": undefined_count,
        "per_sample": per_sample, "_plot_uncertainty": all_uncertainty, "_plot_error": all_error,
    }


def _figures(
    config: dict[str, Any], aggregate: list[dict[str, Any]],
    uncertainty: dict[str, Any], rows: list[dict[str, Any]],
    primary_config: dict[str, Any], expected_dataset: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    output = Path(config["output_directory"]); output.mkdir(parents=True, exist_ok=True); figures = []
    for metric, filename, ylabel in (("rmse", "overall_source_rmse.png", "source RMSE"), ("relative_l2", "overall_relative_l2.png", "relative L2"), ("physics_temperature_rmse", "physics_temperature_rmse.png", "temperature RMSE")):
        means = {method: next(row[metric] for row in aggregate if row["method"] == method and row["test_role"] == "all_test_roles" and row["aggregation_type"] == "pooled_global") for method in METHODS}
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True); ax.bar(list(means), list(means.values())); ax.tick_params(axis="x", rotation=25); ax.set_ylabel(ylabel); ax.set_title(TRUTH_LABEL); fig.savefig(output / filename, dpi=120); plt.close(fig)
        figures.append({"path": str(output / filename), "aggregation_type": "pooled_global", "sample_count": len({row['sample_id'] for row in rows})})
    roles = list(uncertainty["test_role_results"]); values = [uncertainty["test_role_results"][r]["pixel_coverage"] for r in roles]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True); ax.bar(roles, values); ax.axhline(uncertainty["target_coverage"], color="black", linestyle="--"); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("pixel interval coverage"); ax.set_title(TRUTH_LABEL); fig.savefig(output / "interval_coverage_by_test_role.png", dpi=120); plt.close(fig); figures.append({"path": str(output / "interval_coverage_by_test_role.png"), "aggregation_type": "pooled_global_pixels", "sample_count": uncertainty["test_sample_count"]})
    role_method = {(row["test_role"], row["method"]): row["rmse"] for row in aggregate if row.get("test_role") in TEST_ROLES and row["aggregation_type"] == "pooled_global"}
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True); x = np.arange(len(TEST_ROLES)); width = .15
    for index, method in enumerate(METHODS): ax.bar(x + (index - 2) * width, [role_method[(role, method)] for role in TEST_ROLES], width, label=method)
    ax.set_xticks(x, TEST_ROLES, rotation=20); ax.set_ylabel("source RMSE"); ax.set_title(TRUTH_LABEL); ax.legend(fontsize=7); figure_path = output / "per_test_role_comparison.png"; fig.savefig(figure_path, dpi=120); plt.close(fig); figures.append({"path": str(figure_path), "aggregation_type": "pooled_global", "sample_count": len({row['sample_id'] for row in rows})})
    learned = METHODS[:3]; learned_means = [next(row["rmse"] for row in aggregate if row["method"] == method and row["test_role"] == "all_test_roles" and row["aggregation_type"] == "mean_per_sample") for method in learned]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True); ax.bar(learned, learned_means); ax.tick_params(axis="x", rotation=20); ax.set_ylabel("mean per-sample source RMSE"); ax.set_title(TRUTH_LABEL); figure_path = output / "ablation_comparison.png"; fig.savefig(figure_path, dpi=120); plt.close(fig); figures.append({"path": str(figure_path), "aggregation_type": "mean_per_sample", "sample_count": len({row['sample_id'] for row in rows})})
    lookup = {(row["sample_id"], row["method"]): row for row in rows}; sample_ids = sorted({row["sample_id"] for row in rows}); differences = [lookup[(sample_id, "full_residual_attention")]["rmse"] - lookup[(sample_id, "smoothness")]["rmse"] for sample_id in sample_ids]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True); ax.hist(differences, bins=min(30, max(5, len(differences) // 8))); ax.axvline(0, color="black", linestyle="--"); ax.set(xlabel="primary RMSE - smoothness RMSE", ylabel="samples", title=TRUTH_LABEL); figure_path = output / "paired_difference_distributions.png"; fig.savefig(figure_path, dpi=120); plt.close(fig); figures.append({"path": str(figure_path), "aggregation_type": "per_sample", "sample_count": len(differences)})
    family_rows = aggregate_metrics(rows, ("method", "source_family")); families = sorted({row["source_family"] for row in family_rows}); family_values = [next(row["rmse"] for row in family_rows if row["method"] == "full_residual_attention" and row["source_family"] == family and row["aggregation_type"] == "mean_per_sample") for family in families]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True); ax.bar(families, family_values); ax.tick_params(axis="x", rotation=30); ax.set(ylabel="mean per-sample primary source RMSE", title=TRUTH_LABEL); figure_path = output / "source_family_subgroup_comparison.png"; fig.savefig(figure_path, dpi=120); plt.close(fig); figures.append({"path": str(figure_path), "aggregation_type": "mean_per_sample", "sample_count": len({row['sample_id'] for row in rows})})
    uncertainty_values = np.asarray(uncertainty["_plot_uncertainty"]); error_values = np.asarray(uncertainty["_plot_error"]); total_pixels = len(error_values); limit = min(total_pixels, 100000)
    selection = np.linspace(0, total_pixels - 1, limit, dtype=int) if limit else np.array([], int)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True); ax.scatter(uncertainty_values[selection], error_values[selection], s=3, alpha=.25); ax.set(xlabel="MC-dropout predictive standard deviation", ylabel="absolute source error", title=TRUTH_LABEL); figure_path = output / "uncertainty_versus_error.png"; fig.savefig(figure_path, dpi=120); plt.close(fig); figures.append({"path": str(figure_path), "aggregation_type": "pooled_valid_pixels", "sample_count": uncertainty["test_sample_count"], "available_pixel_count": total_pixels, "plotted_pixel_count": limit})
    selected = select_preview_samples(config["dataset_directory"], config["test_roles"], config["preview_sample_count"], config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); probe = SyntheticDatasetReader(config["dataset_directory"]); normalization = probe.normalization; probe.close()
    model = load_trained_model(primary_config, Path(config["checkpoint_directory"]) / "full_residual_attention" / "best.pt", normalization, device, expected_dataset=expected_dataset)
    for preview_index, (role, sample_id) in enumerate(selected):
        dataset = EvaluationDataset(config["dataset_directory"], role, [sample_id]); sample = dataset[0]; dataset.close()
        mean, std = mc_dropout_prediction(model, sample, normalization, device, passes=config["mc_dropout_passes"], seed=config["seed"] + 50000 + preview_index); error = np.abs(mean - sample["true_source"])
        fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
        for axis, image_values, title in zip(axes, (sample["true_source"], mean, error, std), ("true source", "predictive mean", "absolute error", "MC-dropout std")):
            image = axis.imshow(image_values, origin="lower"); axis.set_title(title); fig.colorbar(image, ax=axis, shrink=.7)
        fig.suptitle(f"{TRUTH_LABEL}\n{role}: {sample_id}"); figure_path = output / f"prediction_error_uncertainty_{preview_index + 1:02d}.png"; fig.savefig(figure_path, dpi=120); plt.close(fig); figures.append({"path": str(figure_path), "aggregation_type": "qualitative_sample", "sample_count": 1, "sample_id": sample_id, "test_role": role})
    return figures


def _public_uncertainty(uncertainty: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in uncertainty.items() if not key.startswith("_")}


def _produce_results(
    config: dict[str, Any], partition: dict[str, Any], summaries: dict[str, Any],
    run_configs: dict[str, dict[str, Any]], verification: dict[str, Any], started: float,
) -> dict[str, Any]:
    output = Path(config["output_directory"]); output.mkdir(parents=True, exist_ok=True)
    expected_dataset = verification["dataset_hashes"]
    rows, aggregate = _evaluate(config, run_configs, expected_dataset)
    stratified = []
    for keys in (("method", "test_role", "source_family"), ("method", "test_role", "sensor_strategy"), ("method", "test_role", "noise_level"), ("method", "test_role", "sensor_count_bin")): stratified.extend(aggregate_metrics(rows, keys))
    bootstrap = _bootstrap_outputs(config, rows)
    uncertainty = _uncertainty(config, partition, run_configs["full_residual_attention"], expected_dataset)
    public_uncertainty = _public_uncertainty(uncertainty)
    _write_csv(output / "per_sample_metrics.csv", rows); _write_csv(output / "aggregate_metrics.csv", aggregate); _write_csv(output / "stratified_metrics.csv", stratified); _write_csv(output / "uncertainty_per_sample.csv", uncertainty["per_sample"])
    _atomic_json(output / "paired_bootstrap.json", bootstrap); _atomic_json(output / "uncertainty_calibration.json", public_uncertainty)
    _atomic_json(output / "environment.json", {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}); _atomic_json(output / "run_configuration.json", {k: v for k, v in config.items() if not k.startswith("_")})
    figures = _figures(config, aggregate, uncertainty, rows, run_configs["full_residual_attention"], expected_dataset)
    compatibility = external_compatibility(config["external_manifest"])
    _atomic_json(output / "external_compatibility.json", compatibility)
    dataset_probe = SyntheticDatasetReader(config["dataset_directory"])
    dataset_sample_count = int(dataset_probe.manifest["sample_count"])
    dataset_probe.close()
    overall = [record for record in aggregate if record.get("test_role") == "all_test_roles"]
    summary = {
        "label": TRUTH_LABEL.replace("\n", " - "), "dataset_sample_count": dataset_sample_count,
        "aggregation_definitions": {
            "mean_per_sample": "arithmetic mean of per-sample metrics; used for paired bootstrap, ablation, and subgroup figures",
            "pooled_global": "metrics derived from physical-array sufficient-statistic totals; used for overall and per-role metric figures",
        },
        "overall_metrics": overall, "training_runs": summaries, "checkpoint_verification": verification,
        "test_role_counts": {role: len({r["sample_id"] for r in rows if r["test_role"] == role}) for role in TEST_ROLES},
        "uncertainty": public_uncertainty, "external_compatibility": compatibility, "figures": figures,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(output / "evaluation_summary.json", summary); return summary


def run_all(config: dict[str, Any]) -> dict[str, Any]:
    validate_evaluation_config(config); started = time.perf_counter(); validate_synthetic_dataset(config["dataset_directory"])
    logs = Path(config["log_directory"]); logs.mkdir(parents=True, exist_ok=True)
    partition = partition_validation(config["dataset_directory"], partition_seed=config["partition_seed"], select_count=config["validation_select_count"], calibration_count=config["validation_calibration_count"], output_path=logs / "validation_partition.json")
    summaries, run_configs = _run_model_configs(config, partition)
    verification = verify_checkpoints(config, partition, run_configs, output_path=logs / "checkpoint_verification.json")
    return _produce_results(config, partition, summaries, run_configs, verification, started)


def recompute_results(config: dict[str, Any]) -> dict[str, Any]:
    """Recompute Task 4 artifacts from existing immutable best checkpoints only."""
    validate_evaluation_config(config); started = time.perf_counter(); validate_synthetic_dataset(config["dataset_directory"])
    logs = Path(config["log_directory"]); partition_path = logs / "validation_partition.json"
    try:
        partition = json.loads(partition_path.read_text(encoding="utf-8"))
        summaries = json.loads((logs / "training_runs.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("Existing partition or training-runs record is unavailable") from exc
    run_configs = _compose_run_configs(config)
    verification = verify_checkpoints(config, partition, run_configs, output_path=logs / "checkpoint_verification.json")
    return _produce_results(config, partition, summaries, run_configs, verification, started)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-config", "partition-validation", "train-runs", "evaluate", "uncertainty", "run-all", "recompute-results", "summarize"): commands.add_parser(name).add_argument("configuration", type=Path)
    external = commands.add_parser("external-compatibility"); external.add_argument("manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "external-compatibility": result = external_compatibility(args.manifest)
        else:
            config = load_evaluation_config(args.configuration)
            if args.command == "validate-config": result = {"valid": True, "schema_version": config["schema_version"]}
            elif args.command == "partition-validation": result = partition_validation(config["dataset_directory"], partition_seed=config["partition_seed"], select_count=config["validation_select_count"], calibration_count=config["validation_calibration_count"], output_path=Path(config["log_directory"]) / "validation_partition.json")
            elif args.command == "train-runs":
                partition_path = Path(config["log_directory"]) / "validation_partition.json"
                partition = json.loads(partition_path.read_text(encoding="utf-8"))
                result, _ = _run_model_configs(config, partition)
            elif args.command == "evaluate":
                partition = json.loads((Path(config["log_directory"]) / "validation_partition.json").read_text(encoding="utf-8"))
                run_configs = _compose_run_configs(config)
                verification = verify_checkpoints(config, partition, run_configs, output_path=Path(config["log_directory"]) / "checkpoint_verification.json")
                rows, aggregate = _evaluate(config, run_configs, verification["dataset_hashes"])
                output = Path(config["output_directory"])
                _write_csv(output / "per_sample_metrics.csv", rows)
                _write_csv(output / "aggregate_metrics.csv", aggregate)
                result = {"evaluated_samples": len(rows) // len(METHODS), "methods": list(METHODS)}
            elif args.command == "uncertainty":
                partition = json.loads((Path(config["log_directory"]) / "validation_partition.json").read_text(encoding="utf-8"))
                run_configs = _compose_run_configs(config)
                verification = verify_checkpoints(config, partition, run_configs, output_path=Path(config["log_directory"]) / "checkpoint_verification.json")
                raw_result = _uncertainty(config, partition, run_configs["full_residual_attention"], verification["dataset_hashes"])
                _write_csv(Path(config["output_directory"]) / "uncertainty_per_sample.csv", raw_result["per_sample"])
                result = _public_uncertainty(raw_result)
                _atomic_json(Path(config["output_directory"]) / "uncertainty_calibration.json", result)
            elif args.command == "run-all": result = run_all(config)
            elif args.command == "recompute-results": result = recompute_results(config)
            elif args.command == "summarize": result = json.loads((Path(config["output_directory"]) / "evaluation_summary.json").read_text())
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    except (EvaluationError, DatasetPipelineError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
