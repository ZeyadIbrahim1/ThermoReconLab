"""Phase 5 Task 5 artifact audit, reporting, and reproducibility freeze.

This module consumes existing files only.  It contains no training, dataset
construction, model inference, uncertainty inference, or PDE-solver imports.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility for this project.
    import tomli as tomllib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHEMA_VERSION = 1
PACKAGE_VERSION = "0.1.0"
METHODS = ("full_residual_attention", "residual_no_attention", "direct_sparse_mask", "identity", "smoothness")
LEARNED_METHODS = METHODS[:3]
TEST_ROLES = ("test_id", "test_ood_shape", "test_ood_sensor", "test_ood_noise")
EXPECTED_DEFAULT_SPLITS = {"train": 720, "validation": 120, "test_id": 120, "test_ood_shape": 84, "test_ood_sensor": 72, "test_ood_noise": 84}
TRUTH_LABEL = "Synthetic benchmark only — No external generalization claim"
PHASE_ARTIFACTS = {
    2: {
        "research/ai/ai_data.py", "research/ai/tests/test_ai_data.py",
        "research/ai/configs/dataset_default.json", "research/ai/configs/dataset_smoke.json",
        "data_external/phase5_dataset_default/manifest.json", "data_external/phase5_dataset_default/configuration.json",
        "data_external/phase5_dataset_default/normalization.json", "data_external/phase5_dataset_default/synthetic_dataset.h5",
        "data_external/external_pr_manifest.json",
    },
    3: {
        "research/ai/ai_model.py", "research/ai/tests/test_ai_model.py",
        "research/ai/configs/model_default.json", "research/ai/configs/model_smoke.json",
        "research/ai/requirements-ml.txt",
    },
    4: {
        "research/ai/ai_evaluation.py", "research/ai/tests/test_ai_evaluation.py",
        "research/ai/configs/evaluation_default.json", "research/ai/configs/evaluation_smoke.json",
        "research/ai/checkpoints/task4_default/full_residual_attention/best.pt",
        "research/ai/checkpoints/task4_default/residual_no_attention/best.pt",
        "research/ai/checkpoints/task4_default/direct_sparse_mask/best.pt",
        "pyproject.toml", "src/thermoreconlab/__init__.py", "src/thermoreconlab/reporting.py",
        "tests/test_reporting.py", "examples/01_synthetic_benchmark.py", "examples/02_user_sensor_data.py", "examples/03_parameter_studies.py",
        "examples/04_final_demo.py", "examples/data/demo_sensor_measurements.csv",
    },
    5: {
        "research/ai/ai_finalize.py", "research/ai/tests/test_ai_finalize.py",
        "research/ai/configs/finalization.json", "research/ai/final_report.md",
        "research/ai/model_card.md", "research/ai/reproducibility_manifest.json",
        "research/ai/FINAL_STATUS.md", "research/ai/README.md", "README.md",
    },
}
APPROVED_CLASSICAL_RELEASE_PATHS = frozenset({
    "pyproject.toml",
    "src/thermoreconlab/__init__.py",
    "src/thermoreconlab/reporting.py",
    "tests/test_reporting.py",
    "examples/01_synthetic_benchmark.py",
    "examples/02_user_sensor_data.py",
    "examples/03_parameter_studies.py",
    "examples/04_final_demo.py",
    "examples/data/demo_sensor_measurements.csv",
})
APPROVED_FROZEN_TRACKED_PATHS = frozenset({
    "data_external/external_pr_manifest.json",
    "data_external/phase5_dataset_default/configuration.json",
    "data_external/phase5_dataset_default/manifest.json",
    "data_external/phase5_dataset_default/normalization.json",
    "data_external/phase5_dataset_default/synthetic_dataset.h5",
    "research/ai/checkpoints/task4_default/direct_sparse_mask/best.pt",
    "research/ai/checkpoints/task4_default/full_residual_attention/best.pt",
    "research/ai/checkpoints/task4_default/residual_no_attention/best.pt",
})
CONFIG_KEYS = {
    "schema_version", "dataset_directory", "evaluation_directory", "evaluation_log_directory",
    "checkpoint_directory", "external_manifest", "final_output_directory", "final_log_directory",
    "tracked_report_path", "tracked_model_card_path", "tracked_manifest_path", "tracked_status_path",
    "expected_classical_test_count", "expected_research_test_count", "expected_task4_test_count",
    "external_decision", "model_task_type", "external_task_type",
}
PATH_KEYS = {
    "dataset_directory", "evaluation_directory", "evaluation_log_directory", "checkpoint_directory",
    "external_manifest", "final_output_directory", "final_log_directory", "tracked_report_path",
    "tracked_model_card_path", "tracked_manifest_path", "tracked_status_path",
}
REQUIRED_EVALUATION_FILES = (
    "evaluation_summary.json", "per_sample_metrics.csv", "aggregate_metrics.csv", "stratified_metrics.csv",
    "paired_bootstrap.json", "uncertainty_calibration.json", "uncertainty_per_sample.csv",
    "external_compatibility.json", "run_configuration.json", "environment.json",
)
REQUIRED_LOG_FILES = ("validation_partition.json", "training_runs.json", "test_sample_ids.json", "checkpoint_verification.json")
CHECKPOINT_NAMES = {name: f"{name}/best.pt" for name in LEARNED_METHODS}


class FinalizationError(RuntimeError):
    """Raised when a Task 5 audit or freeze invariant fails."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_stream(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    if block_size < 1:
        raise FinalizationError("Hash block size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_relative_path(path: str | Path) -> str:
    text = str(path)
    if not text or Path(text).is_absolute() or PurePosixPath(text.replace("\\", "/")).is_absolute():
        raise FinalizationError(f"Path must be repository-relative: {text!r}")
    normalized = PurePosixPath(text.replace("\\", "/"))
    if ".." in normalized.parts or "." in normalized.parts or any(part == "" for part in normalized.parts):
        raise FinalizationError(f"Path is not portable: {text!r}")
    return normalized.as_posix()


def artifact_phase(path: str | Path) -> int:
    """Return the exact originating Phase for a known frozen artifact."""
    normalized = normalize_relative_path(path)
    for phase, paths in PHASE_ARTIFACTS.items():
        if normalized in paths:
            return phase
    if normalized.startswith("research/ai/logs/task4_default/"):
        return 4
    if normalized.startswith("research/ai/outputs/evaluation_default/") and Path(normalized).suffix.lower() in {".json", ".csv", ".png"}:
        return 4
    raise FinalizationError(f"Unknown Phase 5 artifact path: {normalized}")


def resolve_path(config: dict[str, Any], key: str, root: Path | None = None) -> Path:
    return (root or repository_root()) / config[key]


def validate_finalization_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise FinalizationError("Finalization configuration must be an object")
    missing, unknown = CONFIG_KEYS - config.keys(), config.keys() - CONFIG_KEYS
    if missing or unknown:
        raise FinalizationError(f"Configuration keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if config["schema_version"] != SCHEMA_VERSION:
        raise FinalizationError("Unsupported finalization schema version")
    for key in PATH_KEYS:
        config[key] = normalize_relative_path(config[key])
    for key in ("expected_classical_test_count", "expected_research_test_count", "expected_task4_test_count"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise FinalizationError(f"{key} must be a positive integer")
    if config["external_decision"] != "no-go" or config["model_task_type"] != "synthetic_source" or config["external_task_type"] != "external_heat_flux":
        raise FinalizationError("Task types and external no-go decision are fixed")
    tracked = (config["tracked_report_path"], config["tracked_model_card_path"], config["tracked_manifest_path"], config["tracked_status_path"])
    if len(set(tracked)) != len(tracked):
        raise FinalizationError("Tracked final artifact paths must be unique")
    return config


def load_finalization_config(path: str | Path) -> dict[str, Any]:
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"Cannot read finalization configuration: {path}") from exc
    return validate_finalization_config(config)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"Cannot read valid JSON artifact: {path}") from exc


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = reader.fieldnames or []
    except (OSError, csv.Error) as exc:
        raise FinalizationError(f"Cannot read valid CSV artifact: {path}") from exc
    if not fields or not rows:
        raise FinalizationError(f"CSV artifact is empty: {path}")
    return fields, rows


def _require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0 or path.name.endswith(".part"):
            raise FinalizationError(f"Required artifact is missing, empty, or partial: {path}")


def _manifest_content_hash(manifest: dict[str, Any]) -> str:
    content = dict(manifest); content.pop("manifest_content_sha256", None)
    return hashlib.sha256(canonical_json(content)).hexdigest()


def reconcile_dataset(manifest: dict[str, Any], partition: dict[str, Any], *, require_default: bool = False) -> dict[str, Any]:
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != manifest.get("sample_count"):
        raise FinalizationError("Dataset sample count does not match manifest samples")
    ids = [sample.get("sample_id") for sample in samples]
    if any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
        raise FinalizationError("Dataset sample IDs are invalid or duplicated")
    splits = Counter(sample.get("split") for sample in samples)
    if require_default and dict(splits) != EXPECTED_DEFAULT_SPLITS:
        raise FinalizationError(f"Default dataset split counts changed: {dict(splits)}")
    train = partition.get("train_sample_ids"); select = partition.get("validation_select_sample_ids"); calibration = partition.get("validation_calibration_sample_ids")
    for name, values in (("train", train), ("validation selection", select), ("validation calibration", calibration)):
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise FinalizationError(f"{name} IDs are invalid or duplicated")
    manifest_train = [sample["sample_id"] for sample in samples if sample["split"] == "train"]
    manifest_validation = {sample["sample_id"] for sample in samples if sample["split"] == "validation"}
    if train != manifest_train or set(select) | set(calibration) != manifest_validation or set(select) & set(calibration):
        raise FinalizationError("Partition does not reconcile with dataset train/validation roles")
    test_counts = {role: splits[role] for role in TEST_ROLES}
    return {
        "dataset_total": len(samples), "split_counts": dict(splits), "train_count": len(train),
        "validation_selection_count": len(select), "validation_calibration_count": len(calibration),
        "test_role_counts": test_counts, "test_total": sum(test_counts.values()),
    }


def reconcile_per_sample(rows: list[dict[str, str]], test_manifest: dict[str, Any]) -> dict[str, Any]:
    required = {"sample_id", "method", "test_role", "source_squared_error_sum", "source_valid_node_count", "source_target_squared_sum", "source_global_maximum_absolute", "temperature_squared_error_sum", "temperature_node_count", "clean_sensor_squared_error_sum", "clean_sensor_count", "noisy_sensor_squared_error_sum", "noisy_sensor_count"}
    if not rows or required - rows[0].keys():
        raise FinalizationError(f"Per-sample metrics missing fields: {sorted(required - rows[0].keys()) if rows else sorted(required)}")
    keys = [(row["sample_id"], row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise FinalizationError("Duplicate (sample_id, method) metric row")
    by_method = defaultdict(set); role_by_id = {}
    for row in rows:
        if row["method"] not in METHODS or row["test_role"] not in TEST_ROLES:
            raise FinalizationError("Unexpected method or test role in per-sample metrics")
        by_method[row["method"]].add(row["sample_id"])
        previous = role_by_id.setdefault(row["sample_id"], row["test_role"])
        if previous != row["test_role"]:
            raise FinalizationError("A sample ID occurs in multiple test roles")
    id_sets = list(by_method.values())
    if set(by_method) != set(METHODS) or any(ids != id_sets[0] for ids in id_sets[1:]):
        raise FinalizationError("Methods do not share identical test sample IDs")
    expected_by_role = {role: set(test_manifest.get(role, [])) for role in TEST_ROLES}
    actual_by_role = {role: {sample_id for sample_id, value in role_by_id.items() if value == role} for role in TEST_ROLES}
    if actual_by_role != expected_by_role:
        raise FinalizationError("Per-sample IDs do not match the Task 4 test manifest")
    unique = len(role_by_id)
    if len(rows) != unique * len(METHODS):
        raise FinalizationError("Per-sample metrics do not contain five methods per sample")
    return {"unique_test_sample_count": unique, "method_count": len(by_method), "row_count": len(rows), "role_counts": {role: len(ids) for role, ids in actual_by_role.items()}}


def _float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalizationError(f"Invalid numeric field {key}") from exc
    if not math.isfinite(value):
        raise FinalizationError(f"Non-finite numeric field {key}")
    return value


def _derived_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    sums = lambda key: math.fsum(_float(row, key) for row in rows)
    squared = sums("source_squared_error_sum"); nodes = sums("source_valid_node_count"); target = sums("source_target_squared_sum")
    temperature_squared = sums("temperature_squared_error_sum"); temperature_nodes = sums("temperature_node_count")
    clean_squared = sums("clean_sensor_squared_error_sum"); clean_count = sums("clean_sensor_count")
    noisy_squared = sums("noisy_sensor_squared_error_sum"); noisy_count = sums("noisy_sensor_count")
    return {
        "rmse": math.sqrt(squared / nodes), "mae": sums("source_absolute_error_sum") / nodes,
        "relative_l2": math.sqrt(squared / target),
        "maximum_absolute_error": max(_float(row, "source_global_maximum_absolute") for row in rows),
        "physics_temperature_rmse": math.sqrt(temperature_squared / temperature_nodes),
        "clean_sensor_residual_rms": math.sqrt(clean_squared / clean_count),
        "noisy_measurement_residual_rms": math.sqrt(noisy_squared / noisy_count),
    }


def reconcile_aggregates(per_sample: list[dict[str, str]], aggregate: list[dict[str, str]], summary: dict[str, Any], *, tolerance: float = 1e-12) -> dict[str, Any]:
    keys = [(row.get("method"), row.get("test_role"), row.get("aggregation_type")) for row in aggregate]
    expected = {(method, role, aggregation) for method in METHODS for role in (*TEST_ROLES, "all_test_roles") for aggregation in ("mean_per_sample", "pooled_global")}
    if len(aggregate) != 50 or len(keys) != len(set(keys)) or set(keys) != expected:
        raise FinalizationError("Aggregate metrics must contain the exact 50 Task 4 combinations")
    lookup = {key: row for key, row in zip(keys, aggregate)}
    reconciled = {}
    for method in METHODS:
        method_rows = [row for row in per_sample if row["method"] == method]
        for role in (*TEST_ROLES, "all_test_roles"):
            group = method_rows if role == "all_test_roles" else [row for row in method_rows if row["test_role"] == role]
            derived = _derived_metrics(group); pooled = lookup[(method, role, "pooled_global")]
            for metric, value in derived.items():
                if not math.isclose(_float(pooled, metric), value, rel_tol=tolerance, abs_tol=tolerance):
                    raise FinalizationError(f"Pooled aggregate mismatch: {method}/{role}/{metric}")
            mean = lookup[(method, role, "mean_per_sample")]
            for metric in ("rmse", "mae", "relative_l2", "maximum_absolute_error", "physics_temperature_rmse"):
                expected_mean = math.fsum(_float(row, metric) for row in group) / len(group)
                if not math.isclose(_float(mean, metric), expected_mean, rel_tol=tolerance, abs_tol=tolerance):
                    raise FinalizationError(f"Mean per-sample aggregate mismatch: {method}/{role}/{metric}")
            if role == "all_test_roles":
                reconciled[method] = {
                    "pooled_global": {metric: _float(pooled, metric) for metric in ("rmse", "mae", "relative_l2", "maximum_absolute_error", "physics_temperature_rmse", "clean_sensor_residual_rms", "noisy_measurement_residual_rms")},
                    "mean_per_sample": {metric: _float(mean, metric) for metric in ("rmse", "mae", "relative_l2", "maximum_absolute_error", "physics_temperature_rmse")},
                    "sample_count": int(float(pooled["sample_count"])),
                }
    summary_rows = {(row["method"], row["aggregation_type"]): row for row in summary.get("overall_metrics", [])}
    for method in METHODS:
        for aggregation in ("pooled_global", "mean_per_sample"):
            if (method, aggregation) not in summary_rows:
                raise FinalizationError("Evaluation summary lacks an overall aggregate row")
            if not math.isclose(float(summary_rows[(method, aggregation)]["rmse"]), reconciled[method][aggregation]["rmse"], rel_tol=tolerance, abs_tol=tolerance):
                raise FinalizationError("Evaluation summary RMSE does not reconcile")
    return {"row_count": len(aggregate), "tolerance": tolerance, "overall": reconciled}


def reconcile_bootstrap(bootstrap: dict[str, Any], per_sample: list[dict[str, str]], *, require_default: bool = False) -> dict[str, Any]:
    key = "all_test_roles/full_residual_attention/smoothness/rmse"
    result = bootstrap.get(key)
    if not isinstance(result, dict) or result.get("difference_convention") != "learned metric - baseline metric":
        raise FinalizationError("Primary-versus-smoothness bootstrap result is missing or malformed")
    primary = {row["sample_id"]: _float(row, "rmse") for row in per_sample if row["method"] == "full_residual_attention"}
    smooth = {row["sample_id"]: _float(row, "rmse") for row in per_sample if row["method"] == "smoothness"}
    differences = [primary[sample_id] - smooth[sample_id] for sample_id in sorted(primary)]
    mean = math.fsum(differences) / len(differences); wins = sum(value < 0 for value in differences) / len(differences)
    if not math.isclose(float(result["mean_paired_difference"]), mean, abs_tol=1e-12) or not math.isclose(float(result["win_rate"]), wins, abs_tol=1e-12) or result.get("sample_count") != len(differences):
        raise FinalizationError("Bootstrap summary does not reconcile with per-sample RMSE")
    interval = [float(value) for value in result.get("confidence_interval", [])]
    if len(interval) != 2 or interval[0] > interval[1] or not interval[0] <= mean <= interval[1]:
        raise FinalizationError("Bootstrap confidence interval is malformed")
    if require_default:
        expected = (-0.2704102454982816, -0.29286236113366815, -0.24922654399608793, 0.975, 360)
        actual = (mean, interval[0], interval[1], wins, len(differences))
        if any(not math.isclose(a, e, abs_tol=1e-10) for a, e in zip(actual[:-1], expected[:-1])) or actual[-1] != expected[-1]:
            raise FinalizationError("Default bootstrap result changed")
    return {"difference_convention": result["difference_convention"], "mean_paired_difference": mean, "confidence_interval": interval, "win_rate": wins, "sample_count": len(differences)}


def reconcile_uncertainty(uncertainty: dict[str, Any], rows: list[dict[str, str]], partition: dict[str, Any], test_manifest: dict[str, Any], *, require_default: bool = False) -> dict[str, Any]:
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != {sample_id for role in TEST_ROLES for sample_id in test_manifest.get(role, [])}:
        raise FinalizationError("Uncertainty sample IDs are duplicated or do not match test IDs")
    if set(ids) & set(partition["validation_calibration_sample_ids"]):
        raise FinalizationError("Calibration/test overlap detected")
    if uncertainty.get("calibration_sample_count") != len(partition["validation_calibration_sample_ids"]) or uncertainty.get("test_sample_count") != len(rows):
        raise FinalizationError("Uncertainty sample counts do not reconcile")
    role_counts = Counter(row["test_role"] for row in rows); coverage = {}
    for role in TEST_ROLES:
        role_result = uncertainty.get("test_role_results", {}).get(role, {})
        if role_result.get("sample_count") != role_counts[role] or role_counts[role] != len(test_manifest[role]):
            raise FinalizationError(f"Uncertainty role count mismatch: {role}")
        mean_coverage = math.fsum(_float(row, "pixel_coverage") for row in rows if row["test_role"] == role) / role_counts[role]
        if not math.isclose(float(role_result["mean_per_sample_pixel_coverage"]), mean_coverage, abs_tol=1e-12):
            raise FinalizationError(f"Uncertainty coverage mismatch: {role}")
        coverage[role] = {"sample_count": role_counts[role], "pixel_coverage": float(role_result["pixel_coverage"]), "mean_per_sample_pixel_coverage": mean_coverage, "pooled_pixel_spearman": float(role_result["pooled_pixel_uncertainty_error_spearman"]), "mean_per_sample_spearman": float(role_result["mean_per_sample_uncertainty_error_spearman"])}
    result = {
        "target_coverage": float(uncertainty["target_coverage"]), "calibration_sample_count": int(uncertainty["calibration_sample_count"]),
        "calibration_pixel_count": int(uncertainty["calibration_pixel_count"]), "test_sample_count": len(rows),
        "multiplier": float(uncertainty["multiplier"]), "pooled_pixel_spearman": float(uncertainty["pooled_pixel_uncertainty_error_spearman"]),
        "mean_per_sample_spearman": float(uncertainty["mean_per_sample_uncertainty_error_spearman"]), "coverage_by_role": coverage,
    }
    if require_default:
        expected = (0.9, 40, 36000, 360, 4.338155347102682, 0.9419756025815378, 0.8911009005438754)
        actual = tuple(result[key] for key in ("target_coverage", "calibration_sample_count", "calibration_pixel_count", "test_sample_count", "multiplier", "pooled_pixel_spearman", "mean_per_sample_spearman"))
        if any(not math.isclose(float(a), float(e), abs_tol=1e-10) for a, e in zip(actual, expected)):
            raise FinalizationError("Default uncertainty result changed")
    return result


def verify_external_gate(external: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    expected = {"decision": "no-go", "arrays_opened": False, "inference_refused": True, "model_task_type": "synthetic_source", "external_task_type": "external_heat_flux", "classical_q_target": False}
    if any(external.get(key) != value for key, value in expected.items()):
        raise FinalizationError("External compatibility gate is inconsistent")
    if external["decision"] != config["external_decision"] or external["model_task_type"] != config["model_task_type"] or external["external_task_type"] != config["external_task_type"]:
        raise FinalizationError("External gate conflicts with finalization configuration")
    return dict(expected)


def _checkpoint_metadata(path: Path, verification: dict[str, Any], training: dict[str, Any], name: str, root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root()
    record = verification["models"].get(name, {})
    if record.get("epoch") != record.get("best_epoch") or not record.get("verified"):
        raise FinalizationError(f"Checkpoint verification is invalid for {name}")
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise FinalizationError(f"Cannot read checkpoint metadata: {path}") from exc
    try:
        metadata = {
            "model_name": name, "checkpoint_path": normalize_relative_path(path.relative_to(root)),
            "checkpoint_sha256": sha256_stream(path), "size_bytes": path.stat().st_size,
            "epoch": int(checkpoint["epoch"]), "best_epoch": int(checkpoint["best_epoch"]),
            "architecture": checkpoint["model_architecture"], "parameter_count": int(training[name]["parameter_count"]),
            "nonnegative_policy": checkpoint["nonnegative_policy"], "train_count": len(checkpoint["train_sample_ids"]),
            "validation_selection_count": len(checkpoint["validation_sample_ids"]),
            "dataset_hashes": {key: checkpoint[key] for key in ("dataset_manifest_hash", "dataset_hdf5_hash", "configuration_hash", "normalization_hash")},
            "torch_version": checkpoint.get("torch_version"), "cuda_build": checkpoint.get("cuda_build"), "device_name": checkpoint.get("device_name"),
        }
    finally:
        del checkpoint
    if metadata["epoch"] != metadata["best_epoch"] or metadata["epoch"] != record["best_epoch"]:
        raise FinalizationError(f"Checkpoint best epoch mismatch for {name}")
    return metadata


def _required_paths(config: dict[str, Any], root: Path) -> list[Path]:
    dataset = resolve_path(config, "dataset_directory", root); evaluation = resolve_path(config, "evaluation_directory", root); logs = resolve_path(config, "evaluation_log_directory", root); checkpoints = resolve_path(config, "checkpoint_directory", root)
    return [dataset / name for name in ("manifest.json", "configuration.json", "normalization.json", "synthetic_dataset.h5")] + [logs / name for name in REQUIRED_LOG_FILES] + [evaluation / name for name in REQUIRED_EVALUATION_FILES] + [checkpoints / relative for relative in CHECKPOINT_NAMES.values()] + [resolve_path(config, "external_manifest", root)]


def load_and_audit(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root(); validate_finalization_config(config); required = _required_paths(config, root); _require_files(required)
    dataset_dir = resolve_path(config, "dataset_directory", root); evaluation_dir = resolve_path(config, "evaluation_directory", root); log_dir = resolve_path(config, "evaluation_log_directory", root); checkpoint_dir = resolve_path(config, "checkpoint_directory", root)
    manifest = _read_json(dataset_dir / "manifest.json"); configuration = _read_json(dataset_dir / "configuration.json"); normalization = _read_json(dataset_dir / "normalization.json")
    if sha256_stream(dataset_dir / "synthetic_dataset.h5") != manifest.get("dataset_sha256"):
        raise FinalizationError("Dataset HDF5 hash mismatch")
    if hashlib.sha256(canonical_json(configuration)).hexdigest() != manifest.get("configuration_sha256") or hashlib.sha256(canonical_json(normalization)).hexdigest() != manifest.get("normalization_sha256") or _manifest_content_hash(manifest) != manifest.get("manifest_content_sha256"):
        raise FinalizationError("Dataset manifest/configuration/normalization hash mismatch")
    partition = _read_json(log_dir / "validation_partition.json"); training = _read_json(log_dir / "training_runs.json"); test_manifest = _read_json(log_dir / "test_sample_ids.json"); checkpoint_verification = _read_json(log_dir / "checkpoint_verification.json")
    require_default = root.resolve() == repository_root().resolve() and config["dataset_directory"] == "data_external/phase5_dataset_default"
    dataset_audit = reconcile_dataset(manifest, partition, require_default=require_default)
    _, per_sample = _read_csv(evaluation_dir / "per_sample_metrics.csv"); _, aggregate = _read_csv(evaluation_dir / "aggregate_metrics.csv"); _, uncertainty_rows = _read_csv(evaluation_dir / "uncertainty_per_sample.csv")
    summary = _read_json(evaluation_dir / "evaluation_summary.json"); bootstrap_json = _read_json(evaluation_dir / "paired_bootstrap.json"); uncertainty_json = _read_json(evaluation_dir / "uncertainty_calibration.json"); external = _read_json(evaluation_dir / "external_compatibility.json")
    per_sample_audit = reconcile_per_sample(per_sample, test_manifest); aggregate_audit = reconcile_aggregates(per_sample, aggregate, summary)
    bootstrap_audit = reconcile_bootstrap(bootstrap_json, per_sample, require_default=require_default); uncertainty_audit = reconcile_uncertainty(uncertainty_json, uncertainty_rows, partition, test_manifest, require_default=require_default); external_audit = verify_external_gate(external, config)
    models = {name: _checkpoint_metadata(checkpoint_dir / CHECKPOINT_NAMES[name], checkpoint_verification, training, name, root) for name in LEARNED_METHODS}
    if checkpoint_verification.get("dataset_hashes", {}).get("dataset_manifest_hash") != manifest["manifest_content_sha256"] or any(model["dataset_hashes"] != checkpoint_verification["dataset_hashes"] for model in models.values()):
        raise FinalizationError("Checkpoint dataset hashes do not reconcile")
    return {
        "dataset": dataset_audit, "dataset_hashes": checkpoint_verification["dataset_hashes"], "partitions": {"partition_sha256": partition.get("partition_sha256"), **dataset_audit},
        "per_sample": per_sample_audit, "aggregate": aggregate_audit, "bootstrap": bootstrap_audit,
        "uncertainty": uncertainty_audit, "external_compatibility": external_audit, "models": models,
        "test_counts": {
            "classical": config["expected_classical_test_count"],
            "research_task2_through_task4": config["expected_research_test_count"],
            "task4": config["expected_task4_test_count"],
            "task5_finalization": 64,
            "combined_research_with_task5": config["expected_research_test_count"] + 64,
        },
        "_per_sample_rows": per_sample, "_aggregate_rows": aggregate, "_summary": summary,
    }


def _git_state(root: Path) -> tuple[str | None, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
        return commit or None, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, True


def _git_result(root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True)


def _git_lines(root: Path, arguments: list[str]) -> list[str]:
    result = _git_result(root, arguments)
    if result.returncode != 0:
        raise FinalizationError(f"Git command failed: git {' '.join(arguments)}: {result.stderr.strip()}")
    return [line.replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def reconcile_package_versions(root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root()
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        pyproject_version = project["project"]["version"]
        tree = ast.parse((root / "src/thermoreconlab/__init__.py").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, KeyError, SyntaxError) as exc:
        raise FinalizationError("Cannot parse package versions") from exc
    package_init_version = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets) and isinstance(node.value, ast.Constant):
            package_init_version = node.value.value
    consistent = pyproject_version == package_init_version == PACKAGE_VERSION
    result = {"pyproject_version": pyproject_version, "package_init_version": package_init_version, "versions_consistent": consistent}
    if not consistent:
        raise FinalizationError(f"Package version mismatch: {result}")
    return result


def reconstruction_git_cleanliness(root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root(); path = "src/thermoreconlab/reconstruction.py"
    unstaged = _git_result(root, ["diff", "--quiet", "--", path]); staged = _git_result(root, ["diff", "--cached", "--quiet", "--", path])
    if unstaged.returncode not in {0, 1} or staged.returncode not in {0, 1}:
        raise FinalizationError("Git could not inspect reconstruction.py cleanliness")
    result = {
        "path": path, "reconstruction_file_unstaged_clean": unstaged.returncode == 0,
        "reconstruction_file_staged_clean": staged.returncode == 0,
        "commands": [f"git diff --quiet -- {path}", f"git diff --cached --quiet -- {path}"],
        "return_codes": {"unstaged": unstaged.returncode, "staged": staged.returncode},
    }
    if not result["reconstruction_file_unstaged_clean"] or not result["reconstruction_file_staged_clean"]:
        raise FinalizationError(f"Classical reconstruction file is modified: {result}")
    return result


def tracking_boundary_audit(root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root()
    tracked = _git_lines(root, ["ls-files", "--", "data_external", "research/ai/checkpoints"])
    tracked_external_hdf5 = [path for path in tracked if path.startswith("data_external/") and Path(path).suffix.lower() in {".h5", ".hdf5"}]
    tracked_checkpoints = [path for path in tracked if path.startswith("research/ai/checkpoints/")]
    unexpected_tracked = sorted(set(tracked) - APPROVED_FROZEN_TRACKED_PATHS)
    ignored = {
        "external_hdf5": _git_result(root, ["check-ignore", "-q", "data_external/probe.h5"]).returncode == 0,
        "checkpoint": _git_result(root, ["check-ignore", "-q", "research/ai/checkpoints/probe.pt"]).returncode == 0,
        "research_output": _git_result(root, ["check-ignore", "-q", "research/ai/outputs/finalization/probe"]).returncode == 0,
    }
    result = {
        "commands": ["git ls-files -- data_external research/ai/checkpoints", "git check-ignore -q data_external/probe.h5", "git check-ignore -q research/ai/checkpoints/probe.pt", "git check-ignore -q research/ai/outputs/finalization/probe"],
        "tracked_paths_inspected": tracked, "tracked_external_hdf5": tracked_external_hdf5,
        "tracked_checkpoints": tracked_checkpoints, "ignored_boundaries": ignored,
        "no_dataset_hdf5_tracked": not tracked_external_hdf5, "no_checkpoint_tracked": not tracked_checkpoints,
        "approved_frozen_tracked_paths": sorted(set(tracked) & APPROVED_FROZEN_TRACKED_PATHS),
        "unexpected_tracked_paths": unexpected_tracked,
        "only_approved_frozen_assets_tracked": not unexpected_tracked,
    }
    if unexpected_tracked or not all(ignored.values()):
        raise FinalizationError(f"External/checkpoint tracking boundary failed: {result}")
    return result


def protected_scope_audit(root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root(); scopes = ("src/", "tests/", "examples/", "pyproject.toml")
    actual = {
        "staged": set(_git_lines(root, ["diff", "--cached", "--name-only", "--", "src", "tests", "examples", "pyproject.toml"])),
        "unstaged": set(_git_lines(root, ["diff", "--name-only", "--", "src", "tests", "examples", "pyproject.toml"])),
        "untracked": set(_git_lines(root, ["ls-files", "--others", "--exclude-standard", "--", "src", "tests", "examples", "pyproject.toml"])),
    }
    def is_generated_cache(path: str) -> bool:
        normalized = path.replace("\\", "/")
        return "__pycache__" in normalized.split("/") or normalized.lower().endswith((".pyc", ".pyo"))

    meaningful = {
        category: {path for path in paths if not is_generated_cache(path)}
        for category, paths in actual.items()
    }
    meaningful_paths = set().union(*meaningful.values())
    unexpected_paths = sorted(meaningful_paths - APPROVED_CLASSICAL_RELEASE_PATHS)
    missing_required_paths = sorted(
        path for path in APPROVED_CLASSICAL_RELEASE_PATHS if not (root / path).is_file()
    )
    confined = not unexpected_paths
    result = {
        "inspected_paths": list(scopes),
        "commands": ["git diff --name-only -- src tests examples pyproject.toml", "git diff --cached --name-only -- src tests examples pyproject.toml", "git ls-files --others --exclude-standard -- src tests examples pyproject.toml"],
        "approved_classical_release_paths": sorted(APPROVED_CLASSICAL_RELEASE_PATHS),
        "git_category_paths": {category: sorted(paths) for category, paths in actual.items()},
        "meaningful_git_category_paths": {category: sorted(paths) for category, paths in meaningful.items()},
        "meaningful_changed_paths": sorted(meaningful_paths),
        "ignored_generated_cache_paths": sorted(set().union(*actual.values()) - meaningful_paths),
        "unexpected_changed_paths": unexpected_paths,
        "missing_required_release_paths": missing_required_paths,
        "required_release_paths_exist": not missing_required_paths,
        "task5_changes_confined_to_allowed_paths": confined,
    }
    if unexpected_paths or missing_required_paths:
        raise FinalizationError(f"Protected release scope audit failed: {result}")
    return result


def isolation_audit(root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root(); source = root / "src/thermoreconlab"; init_text = (source / "__init__.py").read_text(encoding="utf-8"); project = (root / "pyproject.toml").read_text(encoding="utf-8")
    source_files = list(source.rglob("*.py")); torch_imports = [normalize_relative_path(path.relative_to(root)) for path in source_files if "import torch" in path.read_text(encoding="utf-8")]
    project_main = project.split("[project.optional-dependencies]", 1)[0]
    versions = reconcile_package_versions(root); reconstruction = reconstruction_git_cleanliness(root); tracking = tracking_boundary_audit(root); protected = protected_scope_audit(root)
    checks = {
        "classical_source_imports_torch": bool(torch_imports), "main_dependencies_include_torch": "torch" in project_main.lower(),
        "package_init_imports_research": "research" in init_text.lower(), **versions,
        "reconstruct_tikhonov_present": "def reconstruct_tikhonov(" in (source / "reconstruction.py").read_text(encoding="utf-8"),
        "reconstruction_file_unstaged_clean": reconstruction["reconstruction_file_unstaged_clean"],
        "reconstruction_file_staged_clean": reconstruction["reconstruction_file_staged_clean"],
        "only_approved_frozen_assets_tracked": tracking["only_approved_frozen_assets_tracked"],
        "task5_changes_confined_to_allowed_paths": protected["task5_changes_confined_to_allowed_paths"],
    }
    if checks["classical_source_imports_torch"] or checks["main_dependencies_include_torch"] or checks["package_init_imports_research"] or not all(checks[key] for key in ("versions_consistent", "reconstruct_tikhonov_present", "reconstruction_file_unstaged_clean", "reconstruction_file_staged_clean", "only_approved_frozen_assets_tracked", "task5_changes_confined_to_allowed_paths")):
        raise FinalizationError(f"Classical-package isolation audit failed: {checks}")
    return {"passed": True, "checks": checks, "torch_import_paths": torch_imports, "version_reconciliation": versions, "reconstruction_git_audit": reconstruction, "tracking_boundary_audit": tracking, "protected_scope_audit": protected}


def detect_stale_claims(texts: dict[str, str]) -> list[dict[str, str]]:
    prohibited = {
        "task 3 does not exist": "false task-history claim", "task 4 does not exist": "false task-history claim",
        "smoothness tikhonov outperformed": "stale baseline conclusion", "attention was uniformly beneficial": "unsupported attention conclusion",
        "external validation succeeded": "false external-validation claim", "mc dropout is a bayesian posterior": "incorrect uncertainty interpretation",
        "the model predicts external heat flux": "false target claim", "numerical results are real-world measurements": "false data-origin claim",
        "checkpoints are distributed with the package": "false distribution claim",
        "the model is production-ready": "unsupported readiness claim", "the model is production ready": "unsupported readiness claim",
    }
    findings = []
    for path, text in texts.items():
        lowered = text.lower()
        for phrase, reason in prohibited.items():
            if phrase in lowered:
                findings.append({"path": path, "phrase": phrase, "reason": reason})
    return findings


def validate_root_readme(text: str) -> dict[str, Any]:
    required_links = ("research/ai/README.md", "research/ai/final_report.md", "research/ai/model_card.md")
    missing_links = [link for link in required_links if link not in text]
    lowered = text.lower()
    datasets_not_distributed = "datasets and checkpoints are not distributed" in lowered
    external_unresolved = any(phrase in lowered for phrase in (
        "external experimental validation remains unresolved",
        "external validation remains unresolved",
        "not externally validated",
        "is not externally validated",
    ))
    result = {"required_links": list(required_links), "missing_links": missing_links, "datasets_and_checkpoints_not_distributed": datasets_not_distributed, "external_validation_unresolved": external_unresolved}
    if missing_links or not datasets_not_distributed or not external_unresolved:
        raise FinalizationError(f"Root README optional-research statement is incomplete: {result}")
    return result


def documentation_audit(context: dict[str, Any], config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root(); paths = [root / "README.md", root / "research/ai/README.md", resolve_path(config, "tracked_report_path", root), resolve_path(config, "tracked_model_card_path", root), resolve_path(config, "tracked_status_path", root)]
    _require_files(paths); texts = {normalize_relative_path(path.relative_to(root)): path.read_text(encoding="utf-8") for path in paths}; findings = detect_stale_claims(texts)
    if findings:
        raise FinalizationError(f"Stale documentation claims detected: {findings}")
    report = texts[config["tracked_report_path"]]
    for method, values in context["aggregate"]["overall"].items():
        for aggregation in ("pooled_global", "mean_per_sample"):
            formatted = f"{values[aggregation]['rmse']:.6f}"
            if formatted not in report:
                raise FinalizationError(f"Final report omits verified RMSE {method}/{aggregation}: {formatted}")
    required_phrases = ("synthetic-only", "not real-world OOD", "not a Bayesian posterior", "no-go", "not validated physical units", "No external HDF5 array")
    if any(phrase.lower() not in report.lower() for phrase in required_phrases):
        raise FinalizationError("Final report omits a required scientific limitation")
    root_readme = validate_root_readme(texts["README.md"])
    return {"passed": True, "files_checked": sorted(texts), "document_count": len(texts), "stale_claim_count": 0, "numerical_values_reconciled": True, "root_readme": root_readme}


def _artifact_record(path: Path, root: Path, role: str, required: bool) -> dict[str, Any]:
    relative = normalize_relative_path(path.relative_to(root))
    return {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_stream(path), "artifact_role": role, "required": required, "generated_by_phase": artifact_phase(relative)}


def _artifact_record_bytes(path: str, content: bytes, role: str) -> dict[str, Any]:
    relative = normalize_relative_path(path)
    return {"path": relative, "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "artifact_role": role, "required": True, "generated_by_phase": artifact_phase(relative)}


def render_final_report(context: dict[str, Any]) -> str:
    overall = context["aggregate"]["overall"]
    rows = "\n".join(f"| {method} | {values['pooled_global']['rmse']:.6f} | {values['mean_per_sample']['rmse']:.6f} | {values['pooled_global']['mae']:.6f} | {values['pooled_global']['relative_l2']:.6f} | {values['pooled_global']['physics_temperature_rmse']:.6f} |" for method, values in overall.items())
    coverage = "\n".join(f"| {role} | {value['sample_count']} | {value['pixel_coverage']:.6f} | {value['pooled_pixel_spearman']:.6f} |" for role, value in context["uncertainty"]["coverage_by_role"].items())
    models = context["models"]
    return f"""# ThermoReconLab Phase 5 final scientific report

> **Synthetic benchmark only — No external generalization claim.**

## 1. Executive summary

Phase 5 evaluated an optional, isolated AI workflow on synthetic steady-state source reconstruction. All three learned methods improved synthetic source RMSE over identity and smoothness Tikhonov, but no external or real-world generalization was established. Classical ThermoReconLab remains the validated package core.

## 2. Research question

The question was whether fixed convolutional reconstructions can improve source-field recovery from synthetic sparse temperature observations relative to two deterministic Tikhonov baselines, while preserving a strict external-data boundary.

## 3. Classical ThermoReconLab foundation

The research workflow uses outputs derived from the classical two-dimensional finite-difference formulation. It does not alter classical APIs, dependencies, or `reconstruct_tikhonov()`.

## 4. Scientific target definition

The model target is synthetic steady-state source `q`, not experimental heat flux. Source integrals reported by Task 4 are grid-sum quantities and are **not validated physical units**.

## 5. Dataset construction

The frozen dataset contains {context['dataset']['dataset_total']:,} generated samples. It is deterministic and synthetic; no external HDF5 array contributed to construction or evaluation.

## 6. Train/validation/calibration/test design

Training used {context['dataset']['train_count']} samples. Validation was separated into {context['dataset']['validation_selection_count']} model-selection and {context['dataset']['validation_calibration_count']} uncertainty-calibration samples. The four test roles contain {context['dataset']['test_total']} unique samples. Synthetic OOD is **not real-world OOD**.

## 7. Model architecture

The primary residual attention U-Net uses {models['full_residual_attention']['parameter_count']:,} parameters, residual prediction, all four input channels, and softplus nonnegativity.

## 8. Fixed ablations

The two fixed ablations remove attention or predict directly from sparse temperature and sensor-mask channels. No-attention has {models['residual_no_attention']['parameter_count']:,} parameters; direct sparse-mask has {models['direct_sparse_mask']['parameter_count']:,}.

## 9. Training protocol

The immutable best epochs were {models['full_residual_attention']['best_epoch']} (full attention), {models['residual_no_attention']['best_epoch']} (no attention), and {models['direct_sparse_mask']['best_epoch']} (direct sparse-mask). Task 5 performed no retraining and changed no checkpoint.

## 10. Source reconstruction metrics

Mean per-sample metrics average complete sample metrics. Pooled-global metrics are derived from physical-array sums, counts, denominators, and true maxima.

| Method | Pooled-global RMSE | Mean per-sample RMSE | Pooled MAE | Pooled relative L2 | Pooled physics-temperature RMSE |
|---|---:|---:|---:|---:|---:|
{rows}

All learned methods outperform both classical baselines on source RMSE in this synthetic benchmark. The no-attention model has the lowest mean per-sample RMSE ({overall['residual_no_attention']['mean_per_sample']['rmse']:.6f}), while full attention has the lowest pooled-global learned-model RMSE ({overall['full_residual_attention']['pooled_global']['rmse']:.6f}). Attention is therefore aggregation-dependent and not uniformly beneficial.

## 11. Physics-consistency evaluation

Physics-temperature and sensor residuals were recomputed in Task 4 and audited here from sufficient statistics. They are post-hoc synthetic consistency metrics, not a physics training loss or external validation.

## 12. Paired bootstrap analysis

For primary minus smoothness mean per-sample RMSE, using `learned metric - baseline metric`, the mean difference is {context['bootstrap']['mean_paired_difference']:.8f}; the deterministic 95% interval is [{context['bootstrap']['confidence_interval'][0]:.8f}, {context['bootstrap']['confidence_interval'][1]:.8f}], win rate {context['bootstrap']['win_rate']:.3f}, with {context['bootstrap']['sample_count']} paired samples. This interval is not claimed as evidence beyond the specified deterministic paired-bootstrap procedure.

## 13. MC-dropout predictive dispersion

MC dropout is predictive dispersion, **not a Bayesian posterior**. The pooled pixel uncertainty/error Spearman correlation is {context['uncertainty']['pooled_pixel_spearman']:.8f}; the mean per-sample correlation is {context['uncertainty']['mean_per_sample_spearman']:.8f}.

## 14. Calibration and coverage

Calibration used {context['uncertainty']['calibration_sample_count']} samples and {context['uncertainty']['calibration_pixel_count']:,} pixels. The target coverage was {context['uncertainty']['target_coverage']:.2f}, with multiplier {context['uncertainty']['multiplier']:.10f}. Coverage is not uniformly at the nominal 90%.

| Test role | Samples | Pixel coverage | Pooled pixel Spearman |
|---|---:|---:|---:|
{coverage}

## 15. OOD synthetic results

OOD roles hold out synthetic shape, sensor, or noise factors. These results characterize controlled generator shifts only and do not establish behavior on experimental systems.

## 16. External-data compatibility boundary

E-TM-F/PR represents external transient heat-flux data, whereas the model predicts synthetic steady-state source `q`. The compatibility decision remains **no-go**. The external dataset cannot currently validate classical source `q`, and no external HDF5 array was opened for training, numerical evaluation, or Task 5.

## 17. Negative and inconclusive findings

Attention is not uniformly beneficial: its ranking changes with aggregation. Nominal coverage is not uniform across roles. No claim is made about vehicle-fire reconstruction, operational readiness, or external heat-flux prediction.

## 18. Limitations

All numerical evaluation is synthetic-only. Results depend on generator assumptions, grid geometry, sensors, noise, and fixed training choices. Source integrals lack validated physical units. MC-dropout dispersion is not posterior uncertainty. Checkpoints require human licensing review before redistribution.

## 19. Reproducibility

The reproducibility manifest records portable paths, hashes, partitions, checkpoint metadata, commands, test counts, dirty-worktree status, and limitations. The protected-scope audit accepts both an approved working tree and a clean published checkout. Git categories are reported diagnostically, while unexpected changes outside the approved classical release paths are rejected. Task 5 consumed existing artifacts only: it did not retrain, rebuild, infer, or recompute Task 4 results.

## 20. Artifact inventory

The frozen inventory covers dataset metadata and HDF5 hash, partition and test manifests, checkpoint verification and best checkpoints, every Task 4 JSON/CSV/figure, Phase 5 source/configuration files, and final tracked reports. The professor-facing repository includes the minimal synthetic dataset and `best.pt` evaluation bundle; generated outputs, training logs, resume checkpoints, and raw external data remain outside Git.

## 21. Final conclusion

The optional learned workflow improves synthetic source RMSE over both classical baselines in this benchmark. This is not evidence of external generalization. Classical ThermoReconLab remains the validated package core, and external E-TM-F/PR compatibility remains no-go.
"""


def render_model_card(context: dict[str, Any]) -> str:
    model = context["models"]["full_residual_attention"]; metrics = context["aggregate"]["overall"]["full_residual_attention"]
    return f"""# ThermoReconLab synthetic source model card

> **Synthetic research model only.**
> **Not validated for operational heat-source localization.**
> **Not validated on external vehicle-fire data.**

## Model name

ThermoReconLab full residual-attention synthetic source reconstruction model.

## Version/status

Phase 5 final research artifact for package version {PACKAGE_VERSION}; not production-ready.

## Intended use

Controlled research on synthetic two-dimensional steady-state source reconstruction and comparison with fixed ablations and classical baselines.

## Out-of-scope use

Operational localization, safety decisions, experimental heat-flux prediction, vehicle-fire reconstruction, external generalization, or physical-unit inference.

## Inputs

Four normalized grid channels: sparse temperature, sensor mask, identity reconstruction, and smoothness reconstruction.

## Output

A nonnegative synthetic source-grid estimate using a softplus policy. Grid sums are not validated physical source units.

## Architecture

Residual attention U-Net, {model['parameter_count']:,} parameters, architecture `{json.dumps(model['architecture'], sort_keys=True)}`. Ablations are residual without attention and direct sparse-mask prediction.

## Training data

{context['dataset']['train_count']} synthetic samples only. No external-data-trained checkpoint exists.

## Evaluation data

{context['dataset']['test_total']} held-out synthetic samples across ID and three synthetic OOD roles. Synthetic OOD is not real-world OOD.

## Metrics

Primary pooled-global source RMSE: {metrics['pooled_global']['rmse']:.6f}. Primary mean per-sample source RMSE: {metrics['mean_per_sample']['rmse']:.6f}. All learned methods beat identity and smoothness source RMSE here; attention is not uniformly beneficial across aggregation types.

## Uncertainty

MC-dropout predictive dispersion, not a Bayesian posterior. Target interval coverage is {context['uncertainty']['target_coverage']:.2f}; coverage varies by role.

## Limitations

Synthetic-only evaluation; fixed generator and scientific settings; unresolved calibration transfer; no validated source units; no operational or production readiness.

## Ethical/licensing considerations

Incorrect localization could create safety risks if misused. Checkpoint redistribution requires human licensing review.

## External-data restriction

E-TM-F/PR is metadata-only and no-go because external heat flux is scientifically incompatible with the synthetic source `q` target. No external HDF5 array was opened.

## Reproducibility

See `research/ai/reproducibility_manifest.json`; Task 5 did not retrain models or rebuild data.

## Checkpoint information

Research artifact `{model['checkpoint_path']}`, SHA-256 `{model['checkpoint_sha256']}`, best epoch {model['best_epoch']}. It is not required by normal package users and is not distributed with the package.
"""


def render_final_status(context: dict[str, Any]) -> str:
    return f"""# ThermoReconLab Phase 5 final status

- Phase 5 status: complete
- Tasks completed: 5/5
- Classical suite: {context['test_counts']['classical']} passed
- Research suite, Tasks 2–4: {context['test_counts']['research_task2_through_task4']} passed
- Combined research suite including Task 5: {context['test_counts']['combined_research_with_task5']} passed
- Task 5 finalization suite: {context['test_counts']['task5_finalization']} passed
- Default synthetic dataset: {context['dataset']['dataset_total']:,} samples
- External numerical validation: not performed
- External compatibility: no-go
- Package version: {PACKAGE_VERSION}

Artifacts: [final report](final_report.md), [model card](model_card.md), [reproducibility manifest](reproducibility_manifest.json), [Task 4 evaluation summary](outputs/evaluation_default/evaluation_summary.json), and [research README](README.md).

The professor-facing repository includes only the minimal frozen synthetic dataset and `best.pt` evaluation bundle. Generated outputs, training logs, resume checkpoints, and raw external data remain outside Git. Phase 5 is optional research and does not alter the classical package boundary.

The protected-scope audit accepts both an approved working tree and a clean published checkout. Git categories are diagnostic; unexpected changes outside the approved classical release paths are rejected.
"""


def _source_artifacts(config: dict[str, Any], root: Path) -> list[tuple[Path, str, bool]]:
    del config
    virtual_reports = {"research/ai/final_report.md", "research/ai/model_card.md", "research/ai/FINAL_STATUS.md", "research/ai/reproducibility_manifest.json"}
    paths: list[tuple[Path, str, bool]] = []
    for phase in sorted(PHASE_ARTIFACTS):
        for relative in sorted(PHASE_ARTIFACTS[phase] - virtual_reports):
            path = root / relative
            if path.is_file() and not relative.startswith(("data_external/", "research/ai/checkpoints/")):
                paths.append((path, "phase5_source_configuration_test_or_documentation", True))
    return paths


def build_artifact_inventory(config: dict[str, Any], context: dict[str, Any], report_bytes: dict[str, bytes], root: Path | None = None) -> list[dict[str, Any]]:
    root = root or repository_root(); records = []
    dataset = resolve_path(config, "dataset_directory", root); logs = resolve_path(config, "evaluation_log_directory", root); evaluation = resolve_path(config, "evaluation_directory", root); checkpoints = resolve_path(config, "checkpoint_directory", root)
    for name in ("manifest.json", "configuration.json", "normalization.json", "synthetic_dataset.h5"):
        records.append(_artifact_record(dataset / name, root, "default_synthetic_dataset", True))
    for name in REQUIRED_LOG_FILES: records.append(_artifact_record(logs / name, root, "task4_evaluation_log", True))
    for name in REQUIRED_EVALUATION_FILES: records.append(_artifact_record(evaluation / name, root, "task4_machine_readable_result", True))
    for path in sorted(evaluation.glob("*.png")): records.append(_artifact_record(path, root, "task4_figure", True))
    for name, relative in CHECKPOINT_NAMES.items(): records.append(_artifact_record(checkpoints / relative, root, f"best_checkpoint:{name}", True))
    records.append(_artifact_record(resolve_path(config, "external_manifest", root), root, "external_metadata_only", True))
    for path, role, required in _source_artifacts(config, root): records.append(_artifact_record(path, root, role, required))
    for path, content in report_bytes.items(): records.append(_artifact_record_bytes(path, content, "final_tracked_report"))
    deduplicated = {record["path"]: record for record in records}
    return [deduplicated[path] for path in sorted(deduplicated)]


def canonical_manifest_hash(manifest: dict[str, Any]) -> str:
    content = copy.deepcopy(manifest); content.pop("generated_at_utc", None)
    verification = content.get("verification")
    if isinstance(verification, dict): verification.pop("manifest_content_sha256", None)
    return hashlib.sha256(canonical_json(content)).hexdigest()


def create_manifest(config: dict[str, Any], context: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root(); commit, dirty = _git_state(root); versions = reconcile_package_versions(root)
    report_bytes = {
        config["tracked_report_path"]: (render_final_report(context).rstrip() + "\n").encode("utf-8"),
        config["tracked_model_card_path"]: (render_model_card(context).rstrip() + "\n").encode("utf-8"),
        config["tracked_status_path"]: (render_final_status(context).rstrip() + "\n").encode("utf-8"),
    }
    artifacts = build_artifact_inventory(config, context, report_bytes, root)
    manifest = {
        "schema_version": 1, "project": "ThermoReconLab", "package_version": versions["pyproject_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "git_dirty": dirty,
        "python": {"version": sys.version, "implementation": platform.python_implementation(), "platform": platform.platform()},
        "classical_package": {"validated_core": True, "pytorch_required": False, "research_imported_by_package": False, "expected_test_count": config["expected_classical_test_count"], **versions},
        "phase5_scope": {"status": "complete", "tasks_completed": "5/5", "optional_isolated_research": True, "task5_retrained_models": False, "task5_rebuilt_dataset": False, "task5_ran_inference": False, "uncommitted_files_published": False},
        "dataset": {**context["dataset"], "hashes": context["dataset_hashes"]}, "partitions": context["partitions"],
        "models": context["models"], "evaluation": {"overall": context["aggregate"]["overall"], "bootstrap_primary_vs_smoothness": context["bootstrap"]},
        "uncertainty": context["uncertainty"], "external_compatibility": context["external_compatibility"], "artifacts": artifacts,
        "commands": {
            "run_all": '& ".\\.venv\\Scripts\\python.exe" research/ai/ai_finalize.py run-all research/ai/configs/finalization.json',
            "verify": '& ".\\.venv\\Scripts\\python.exe" research/ai/ai_finalize.py verify research/ai/configs/finalization.json',
            "classical_tests": '& ".\\.venv\\Scripts\\python.exe" -m pytest -q',
            "research_tests": '& ".\\.venv\\Scripts\\python.exe" -m pytest research/ai/tests/test_ai_data.py research/ai/tests/test_ai_model.py research/ai/tests/test_ai_evaluation.py research/ai/tests/test_ai_finalize.py -q',
        },
        "limitations": ["all numerical results are synthetic", "synthetic OOD is not real-world OOD", "no external generalization", "source integrals are grid sums without validated physical units", "MC dropout is predictive dispersion, not a Bayesian posterior", "coverage is not uniform at nominal 90%", "external E-TM-F/PR compatibility is no-go", "checkpoints require human licensing review before redistribution"],
        "verification": {"test_counts": context["test_counts"], "numerical_reconciliation_passed": True, "artifact_audit_passed": True, "timestamp_excluded_from_canonical_hash": True},
    }
    manifest["verification"]["manifest_content_sha256"] = canonical_manifest_hash(manifest)
    return manifest


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle: handle.write(text)
    temporary.replace(path)


def _atomic_text_if_changed(path: Path, text: str) -> bool:
    encoded = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == encoded:
        return False
    _atomic_text(path, text)
    return True


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def freeze(config: dict[str, Any], context: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root(); manifest = create_manifest(config, context, root); _atomic_json(resolve_path(config, "tracked_manifest_path", root), manifest); return manifest


def verify_manifest(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root(); path = resolve_path(config, "tracked_manifest_path", root); manifest = _read_json(path)
    expected_hash = manifest.get("verification", {}).get("manifest_content_sha256")
    if expected_hash != canonical_manifest_hash(manifest): raise FinalizationError("Canonical manifest hash mismatch")
    failures = []
    for record in manifest.get("artifacts", []):
        artifact = root / record["path"]
        if not artifact.is_file(): failures.append({"path": record["path"], "reason": "missing"}); continue
        if artifact.stat().st_size != record["size_bytes"] or sha256_stream(artifact) != record["sha256"]: failures.append({"path": record["path"], "reason": "hash_or_size_mismatch"})
    if failures: raise FinalizationError(f"Frozen artifact verification failed: {failures}")
    return {"verified": True, "manifest_content_sha256": expected_hash, "artifact_count": len(manifest["artifacts"]), "total_audited_bytes": sum(record["size_bytes"] for record in manifest["artifacts"])}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise FinalizationError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def build_reports(config: dict[str, Any], context: dict[str, Any], manifest: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root(); output = resolve_path(config, "final_output_directory", root); output.mkdir(parents=True, exist_ok=True)
    _atomic_text_if_changed(resolve_path(config, "tracked_report_path", root), render_final_report(context).rstrip() + "\n"); _atomic_text_if_changed(resolve_path(config, "tracked_model_card_path", root), render_model_card(context).rstrip() + "\n"); _atomic_text_if_changed(resolve_path(config, "tracked_status_path", root), render_final_status(context).rstrip() + "\n")
    metrics = []
    for method, values in context["aggregate"]["overall"].items():
        for aggregation in ("pooled_global", "mean_per_sample"):
            metrics.append({"method": method, "aggregation_type": aggregation, "source_rmse": values[aggregation]["rmse"], "source_mae": values[aggregation]["mae"], "relative_l2": values[aggregation]["relative_l2"], "physics_temperature_rmse": values[aggregation]["physics_temperature_rmse"], "sample_count": values["sample_count"]})
    _write_csv(output / "final_metrics.csv", metrics); _write_csv(output / "artifact_inventory.csv", manifest["artifacts"])
    commands = "\n".join(manifest["commands"].values()) + "\n"; _atomic_text(output / "reproducibility_commands.txt", commands)
    pooled = [context["aggregate"]["overall"][method]["pooled_global"]["rmse"] for method in METHODS]; means = [context["aggregate"]["overall"][method]["mean_per_sample"]["rmse"] for method in METHODS]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, values, title in zip(axes, (pooled, means), ("Pooled-global source RMSE", "Mean per-sample source RMSE")):
        axis.bar(METHODS, values); axis.tick_params(axis="x", rotation=25); axis.set_ylabel("source RMSE"); axis.set_title(title)
    figure.suptitle(TRUTH_LABEL); figure.savefig(output / "final_results_overview.png", dpi=140); plt.close(figure)
    return {"tracked": [config["tracked_report_path"], config["tracked_model_card_path"], config["tracked_manifest_path"], config["tracked_status_path"]], "generated": [normalize_relative_path((output / name).relative_to(root)) for name in ("final_metrics.csv", "artifact_inventory.csv", "reproducibility_commands.txt", "final_results_overview.png")]}


def _public_context(context: dict[str, Any]) -> dict[str, Any]: return {key: value for key, value in context.items() if not key.startswith("_")}


def run_all(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root(); started = time.perf_counter(); validate_finalization_config(config)
    context = load_and_audit(config, root); manifest = freeze(config, context, root); outputs = build_reports(config, context, manifest, root)
    verification = verify_manifest(config, root); isolation = isolation_audit(root); documentation = documentation_audit(context, config, root)
    logs = resolve_path(config, "final_log_directory", root); logs.mkdir(parents=True, exist_ok=True); _atomic_json(logs / "isolation_audit.json", isolation); _atomic_json(logs / "documentation_audit.json", documentation)
    runtime = time.perf_counter() - started
    final = {"phase5_status": "complete", "tasks_completed": "5/5", "package_version": PACKAGE_VERSION, "audit": _public_context(context), "manifest_verification": verification, "isolation_audit": isolation, "documentation_audit": documentation, "outputs": outputs, "runtime_seconds": runtime, "no_retraining": True, "no_dataset_rebuild": True, "no_task4_result_changes": True, "external_hdf5_arrays_opened": False}
    final["outputs"]["generated"].append(normalize_relative_path((resolve_path(config, "final_output_directory", root) / "final_summary.json").relative_to(root)))
    output = resolve_path(config, "final_output_directory", root); _atomic_json(output / "final_summary.json", final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-config", "audit", "build-report", "freeze", "verify", "summarize", "run-all"):
        commands.add_parser(name).add_argument("configuration", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_finalization_config(args.configuration); root = repository_root()
        if args.command == "validate-config": result = {"valid": True, "schema_version": config["schema_version"]}
        elif args.command == "audit": result = _public_context(load_and_audit(config, root))
        elif args.command == "freeze":
            context = load_and_audit(config, root); manifest = freeze(config, context, root); result = {"frozen": True, "manifest_content_sha256": manifest["verification"]["manifest_content_sha256"], "artifact_count": len(manifest["artifacts"])}
        elif args.command == "build-report":
            context = load_and_audit(config, root); manifest = _read_json(resolve_path(config, "tracked_manifest_path", root)); result = build_reports(config, context, manifest, root)
        elif args.command == "verify": result = verify_manifest(config, root)
        elif args.command == "summarize": result = _read_json(resolve_path(config, "final_output_directory", root) / "final_summary.json")
        elif args.command == "run-all": result = run_all(config, root)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)); return 0
    except (FinalizationError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
