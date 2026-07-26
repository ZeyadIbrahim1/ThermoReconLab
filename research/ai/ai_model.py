"""Isolated Task 3 residual-attention U-Net and reproducible training system."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_data import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    DatasetPipelineError,
    SyntheticDatasetReader,
    _canonical_json,
)


MODEL_CONFIG_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 2
MINIMUM_TORCH_VERSION = (2, 1)
ALLOWED_TRAINING_SPLITS = ("train", "validation")


class ModelConfigurationError(ValueError):
    """Raised when a Task 3 configuration is invalid."""


class TrainingError(RuntimeError):
    """Raised when training encounters invalid or non-finite state."""


def _version_tuple(version: str) -> tuple[int, int]:
    core = version.split("+", 1)[0].split(".")
    return int(core[0]), int(core[1])


def require_supported_torch() -> None:
    if _version_tuple(torch.__version__) < MINIMUM_TORCH_VERSION:
        raise RuntimeError(
            f"PyTorch >= {MINIMUM_TORCH_VERSION[0]}.{MINIMUM_TORCH_VERSION[1]} is required"
        )


def load_model_config(path: str | Path) -> dict[str, Any]:
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelConfigurationError(f"Cannot read model configuration: {path}") from exc
    return validate_model_config(config)


def validate_model_config(config: dict[str, Any]) -> dict[str, Any]:
    config.setdefault("sample_selection_manifest", None)
    required = {
        "schema_version", "run_label", "seed", "dataset_directory",
        "output_directory", "checkpoint_directory", "log_directory",
        "device_policy", "architecture", "nonnegative_policy", "batch_size",
        "workers", "epochs", "optimizer", "scheduler", "loss_weights",
        "mixed_precision", "gradient_clip_norm", "early_stopping",
        "history_format", "resume", "torch_compile", "functional_only",
    }
    if not isinstance(config, dict) or required - config.keys():
        raise ModelConfigurationError(
            f"Missing model configuration keys: {sorted(required - config.keys())}"
        )
    if config["schema_version"] != MODEL_CONFIG_SCHEMA_VERSION:
        raise ModelConfigurationError("Unsupported model configuration schema")
    for key in ("seed", "batch_size", "workers", "epochs"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ModelConfigurationError(f"{key} must be an integer")
    if config["batch_size"] < 1 or config["workers"] < 0 or config["epochs"] < 1:
        raise ModelConfigurationError("batch_size/epochs must be positive and workers nonnegative")
    if config["device_policy"] not in {"auto", "cpu", "cuda"}:
        raise ModelConfigurationError("device_policy must be auto, cpu, or cuda")
    architecture = config["architecture"]
    architecture.setdefault("attention", True)
    architecture.setdefault("prediction_mode", "residual")
    architecture.setdefault("input_channel_mask", [1, 1, 1, 1])
    expected_arch = {"input_channels", "output_channels", "base_channels", "depth", "dropout", "upsampling"}
    if not isinstance(architecture, dict) or expected_arch - architecture.keys():
        raise ModelConfigurationError("Architecture configuration is incomplete")
    if architecture["input_channels"] != 4 or architecture["output_channels"] != 1:
        raise ModelConfigurationError("Architecture requires exactly four inputs and one output")
    for key in ("base_channels", "depth"):
        if isinstance(architecture[key], bool) or not isinstance(architecture[key], int) or architecture[key] < 1:
            raise ModelConfigurationError(f"architecture.{key} must be a positive integer")
    if architecture["depth"] > 5:
        raise ModelConfigurationError("architecture.depth must not exceed 5")
    dropout = architecture["dropout"]
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise ModelConfigurationError("architecture.dropout must be in [0, 1)")
    if architecture["upsampling"] not in {"bilinear", "transpose"}:
        raise ModelConfigurationError("architecture.upsampling must be bilinear or transpose")
    if not isinstance(architecture["attention"], bool):
        raise ModelConfigurationError("architecture.attention must be Boolean")
    if architecture["prediction_mode"] not in {"residual", "direct"}:
        raise ModelConfigurationError("architecture.prediction_mode must be residual or direct")
    channel_mask = architecture["input_channel_mask"]
    if (
        not isinstance(channel_mask, list) or len(channel_mask) != 4
        or any(value not in (0, 1, False, True) for value in channel_mask)
        or not any(bool(value) for value in channel_mask)
    ):
        raise ModelConfigurationError("architecture.input_channel_mask must contain four binary values and not be all zero")
    if config["nonnegative_policy"] not in {"none", "relu", "softplus"}:
        raise ModelConfigurationError("Invalid nonnegative output policy")
    optimizer = config["optimizer"]
    if optimizer.get("name") != "AdamW" or optimizer.get("learning_rate", 0) <= 0 or optimizer.get("weight_decay", -1) < 0:
        raise ModelConfigurationError("Invalid AdamW configuration")
    if config["scheduler"].get("name") not in {"none", "cosine"}:
        raise ModelConfigurationError("scheduler.name must be none or cosine")
    validate_loss_weights(config["loss_weights"])
    if not isinstance(config["mixed_precision"], bool) or not isinstance(config["torch_compile"], bool):
        raise ModelConfigurationError("mixed_precision and torch_compile must be Boolean")
    if config["torch_compile"]:
        raise ModelConfigurationError("torch_compile is disabled for Task 3")
    clip = config["gradient_clip_norm"]
    if not isinstance(clip, (int, float)) or isinstance(clip, bool) or not math.isfinite(clip) or clip <= 0:
        raise ModelConfigurationError("gradient_clip_norm must be positive and finite")
    early = config["early_stopping"]
    if early.get("patience", 0) < 1 or early.get("minimum_improvement", -1) < 0:
        raise ModelConfigurationError("Invalid early-stopping configuration")
    if config["history_format"] != "jsonl":
        raise ModelConfigurationError("Only JSONL history is supported")
    return config


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
        )
        self.skip = (
            nn.Identity() if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )
        self.activation = nn.SiLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 8, 1)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
            nn.SiLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs * self.channel(inputs)
        spatial = torch.cat((values.mean(1, keepdim=True), values.amax(1, keepdim=True)), dim=1)
        return values * self.spatial(spatial)


class ResidualAttentionUNet(nn.Module):
    """Compact U-Net whose one-channel head predicts a normalized residual."""

    def __init__(
        self, *, input_channels: int = 4, output_channels: int = 1,
        base_channels: int = 32, depth: int = 3, dropout: float = 0.1,
        upsampling: str = "bilinear", nonnegative_policy: str = "softplus",
        source_zero_normalized: float = 0.0,
        attention: bool = True, prediction_mode: str = "residual",
        input_channel_mask: list[int | bool] | tuple[int | bool, ...] = (1, 1, 1, 1),
    ):
        super().__init__()
        if input_channels != 4 or output_channels != 1 or depth < 1:
            raise ValueError("Model requires four inputs, one output, and positive depth")
        if upsampling not in {"bilinear", "transpose"}:
            raise ValueError("upsampling must be bilinear or transpose")
        if nonnegative_policy not in {"none", "relu", "softplus"}:
            raise ValueError("Invalid nonnegative policy")
        if prediction_mode not in {"residual", "direct"}:
            raise ValueError("prediction_mode must be residual or direct")
        if len(input_channel_mask) != 4 or not any(bool(value) for value in input_channel_mask):
            raise ValueError("input_channel_mask must contain four values and not be all zero")
        if isinstance(source_zero_normalized, bool) or not isinstance(
            source_zero_normalized, (int, float)
        ) or not math.isfinite(source_zero_normalized):
            raise ValueError("source_zero_normalized must be a finite scalar")
        self.depth = depth
        self.nonnegative_policy = nonnegative_policy
        self.prediction_mode = prediction_mode
        self.attention_enabled = attention
        self.register_buffer(
            "input_channel_mask",
            torch.tensor([bool(value) for value in input_channel_mask], dtype=torch.float32).view(1, 4, 1, 1),
        )
        self.register_buffer(
            "source_zero_normalized",
            torch.tensor(float(source_zero_normalized), dtype=torch.float32),
        )
        channels = [base_channels * 2**level for level in range(depth + 1)]
        self.encoders = nn.ModuleList()
        previous = input_channels
        for channel_count in channels[:-1]:
            self.encoders.append(nn.Sequential(
                ResidualBlock(previous, channel_count, dropout),
                ChannelSpatialAttention(channel_count) if attention else nn.Identity(),
            ))
            previous = channel_count
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-2], channels[-1], dropout),
            ChannelSpatialAttention(channels[-1]) if attention else nn.Identity(),
        )
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            if upsampling == "transpose":
                self.upsamplers.append(nn.ConvTranspose2d(current, skip_channels, 2, stride=2))
                up_channels = skip_channels
            else:
                self.upsamplers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(current, skip_channels, 1),
                ))
                up_channels = skip_channels
            self.decoders.append(nn.Sequential(
                ResidualBlock(up_channels + skip_channels, skip_channels, dropout),
                ChannelSpatialAttention(skip_channels) if attention else nn.Identity(),
            ))
            current = skip_channels
        self.residual_head = nn.Conv2d(current, output_channels, 1)

    def forward(self, inputs: Tensor, smoothness_normalized: Tensor, valid_mask: Tensor) -> Tensor:
        divisor = 2**self.depth
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("inputs must have shape (batch, 4, height, width)")
        expected = (inputs.shape[0], 1, inputs.shape[2], inputs.shape[3])
        if tuple(smoothness_normalized.shape) != expected:
            raise ValueError(
                f"smoothness_normalized must have shape {expected}, "
                f"received {tuple(smoothness_normalized.shape)}"
            )
        if tuple(valid_mask.shape) != expected:
            raise ValueError(
                f"valid_mask must have shape {expected}, received {tuple(valid_mask.shape)}"
            )
        if self.source_zero_normalized.ndim != 0:
            raise ValueError("source_zero_normalized must be scalar")
        if inputs.shape[-2] % divisor or inputs.shape[-1] % divisor:
            raise ValueError(f"Spatial dimensions must be divisible by {divisor}")
        skips = []
        values = inputs * self.input_channel_mask.to(device=inputs.device, dtype=inputs.dtype)
        for encoder in self.encoders:
            values = encoder(values)
            skips.append(values)
            values = self.pool(values)
        values = self.bottleneck(values)
        for upsample, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            values = upsample(values)
            if values.shape[-2:] != skip.shape[-2:]:
                raise ValueError("Decoder and skip spatial shapes are incompatible")
            values = decoder(torch.cat((values, skip), dim=1))
        head = self.residual_head(values)
        raw_prediction = smoothness_normalized + head if self.prediction_mode == "residual" else head
        threshold = self.source_zero_normalized.to(
            device=raw_prediction.device, dtype=raw_prediction.dtype
        )
        if self.nonnegative_policy == "relu":
            prediction = torch.maximum(raw_prediction, threshold)
        elif self.nonnegative_policy == "softplus":
            prediction = threshold + torch.nn.functional.softplus(
                raw_prediction - threshold
            )
        else:
            prediction = raw_prediction
        return prediction * valid_mask + threshold * (1.0 - valid_mask)


def initialize_model(model: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def source_zero_from_normalization(normalization: dict[str, Any]) -> float:
    stats = normalization["statistics"]["true_source"]
    scale = float(stats["scale"])
    mean = float(stats["mean"])
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise DatasetPipelineError("Invalid persisted true_source normalization")
    return (0.0 - mean) / scale


def build_model(
    config: dict[str, Any], normalization: dict[str, Any] | None = None
) -> ResidualAttentionUNet:
    architecture = config["architecture"]
    model = ResidualAttentionUNet(
        **architecture, nonnegative_policy=config["nonnegative_policy"],
        source_zero_normalized=(
            source_zero_from_normalization(normalization)
            if normalization is not None else 0.0
        ),
    )
    initialize_model(model, config["seed"])
    if parameter_count(model) > 5_000_000:
        raise ModelConfigurationError("Model exceeds the 5,000,000 parameter ceiling")
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class SyntheticTorchDataset(Dataset):
    """Lazy Task 2 adapter that reuses persisted training normalization."""

    def __init__(
        self, dataset_directory: str | Path, split: str,
        sample_ids: list[str] | None = None,
    ):
        if split not in ALLOWED_TRAINING_SPLITS:
            raise ValueError("Task 3 dataset split must be train or validation")
        self.directory = Path(dataset_directory)
        probe = SyntheticDatasetReader(self.directory)
        if any(sample.get("task_type") != "synthetic_source" for sample in probe.manifest["samples"]):
            raise DatasetPipelineError("Task 3 accepts only synthetic_source datasets")
        split_indices = [
            index for index, sample in enumerate(probe.manifest["samples"])
            if sample["split"] == split
        ]
        if sample_ids is not None:
            if len(sample_ids) != len(set(sample_ids)):
                probe.close()
                raise DatasetPipelineError(f"Duplicate sample ID in {split} selection")
            available = {probe.manifest["samples"][index]["sample_id"]: index for index in split_indices}
            unknown = sorted(set(sample_ids) - set(available))
            if unknown:
                probe.close()
                raise DatasetPipelineError(f"Unknown or wrong-split {split} sample IDs: {unknown}")
            self.indices = [available[sample_id] for sample_id in sample_ids]
        else:
            self.indices = split_indices
        self.sample_ids = [probe.manifest["samples"][index]["sample_id"] for index in self.indices]
        self.normalization = json.loads(json.dumps(probe.normalization))
        self.manifest = json.loads(json.dumps(probe.manifest))
        probe.close()
        if not self.indices:
            raise DatasetPipelineError(f"Synthetic split is empty: {split}")
        if self.normalization.get("method") != "global_standard" or self.normalization.get("fitted_split") != "train":
            raise DatasetPipelineError("Persisted training normalization is required")
        self.split = split
        self._reader: SyntheticDatasetReader | None = None

    def _get_reader(self) -> SyntheticDatasetReader:
        if self._reader is None:
            self._reader = SyntheticDatasetReader(self.directory)
        return self._reader

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self) -> "SyntheticTorchDataset":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_reader"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def _normalize(self, values: np.ndarray, field: str) -> np.ndarray:
        stats = self.normalization["statistics"][field]
        return (values.astype(np.float32) - stats["mean"]) / stats["scale"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._get_reader()[self.indices[index]]
        sparse = self._normalize(sample["sparse_temperature"], "sparse_temperature")
        identity = self._normalize(sample["identity_reconstruction"], "true_source")
        smoothness = self._normalize(sample["smoothness_reconstruction"], "true_source")
        target = self._normalize(sample["true_source"], "true_source")
        mask = sample["sensor_mask"].astype(np.float32)
        valid = sample["source_valid_mask"].astype(np.float32)
        inputs = np.stack((sparse, mask, identity, smoothness), axis=0)
        return {
            "inputs": torch.from_numpy(inputs.copy()),
            "smoothness_normalized": torch.from_numpy(smoothness[None].copy()),
            "target_normalized": torch.from_numpy(target[None].copy()),
            "valid_mask": torch.from_numpy(valid[None].copy()),
            "target_physical": torch.from_numpy(sample["true_source"][None].astype(np.float32).copy()),
            "identity_physical": torch.from_numpy(sample["identity_reconstruction"][None].astype(np.float32).copy()),
            "smoothness_physical": torch.from_numpy(sample["smoothness_reconstruction"][None].astype(np.float32).copy()),
            "sample_id": sample["sample_id"], "split": sample["split"],
        }


def make_loaders(
    dataset_directory: str | Path, *, batch_size: int, workers: int,
    seed: int, pin_memory: bool,
    train_sample_ids: list[str] | None = None,
    validation_sample_ids: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, SyntheticTorchDataset, SyntheticTorchDataset]:
    train_dataset = SyntheticTorchDataset(dataset_directory, "train", train_sample_ids)
    validation_dataset = SyntheticTorchDataset(dataset_directory, "validation", validation_sample_ids)
    overlap = set(train_dataset.sample_ids) & set(validation_dataset.sample_ids)
    if overlap:
        raise DatasetPipelineError(f"Train/validation sample IDs overlap: {sorted(overlap)}")
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size, "num_workers": workers,
        "pin_memory": pin_memory, "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator,
        worker_init_fn=_seed_worker if workers else None, **common,
    )
    train_loader.task3_generator = generator  # type: ignore[attr-defined]
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader, train_dataset, validation_dataset


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_loss_weights(weights: dict[str, Any]) -> dict[str, float]:
    if set(weights) != {"mse", "l1", "gradient"}:
        raise ModelConfigurationError("Loss weights must contain mse, l1, and gradient")
    values = {}
    for key, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ModelConfigurationError(f"Loss weight {key} must be finite and nonnegative")
        values[key] = float(value)
    if not any(values.values()):
        raise ModelConfigurationError("At least one loss weight must be positive")
    return values


def masked_loss(
    prediction: Tensor, target: Tensor, mask: Tensor, weights: dict[str, Any]
) -> tuple[Tensor, dict[str, Tensor]]:
    weight_values = validate_loss_weights(weights)
    valid_count = mask.sum()
    if valid_count <= 0:
        raise TrainingError("Loss mask contains no valid nodes")
    difference = prediction - target
    zero = prediction.new_zeros(())
    mse = ((difference.square() * mask).sum() / valid_count) if weight_values["mse"] else zero
    l1 = ((difference.abs() * mask).sum() / valid_count) if weight_values["l1"] else zero
    gradient = zero
    if weight_values["gradient"]:
        dx_mask = mask[..., 1:, :] * mask[..., :-1, :]
        dy_mask = mask[..., :, 1:] * mask[..., :, :-1]
        dx = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
        dy = (prediction[..., :, 1:] - prediction[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
        denominator = dx_mask.sum() + dy_mask.sum()
        gradient = (
            (dx.abs() * dx_mask).sum() + (dy.abs() * dy_mask).sum()
        ) / denominator.clamp_min(1)
    total = weight_values["mse"] * mse + weight_values["l1"] * l1 + weight_values["gradient"] * gradient
    return total, {"mse": mse, "l1": l1, "gradient": gradient, "total": total}


def physical_metrics(prediction: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
    difference = (prediction - target) * mask
    count = mask.sum().clamp_min(1)
    squared = difference.square().sum()
    target_norm = (target * mask).square().sum().sqrt()
    return {
        "rmse": float((squared / count).sqrt().item()),
        "mae": float(difference.abs().sum().div(count).item()),
        "relative_l2": float(squared.sqrt().div(target_norm.clamp_min(torch.finfo(target.dtype).eps)).item()),
        "maximum_absolute": float(difference.abs().amax().item()),
    }


def denormalize_source(
    values: Tensor, normalization: dict[str, Any], valid_mask: Tensor | None = None
) -> Tensor:
    stats = normalization["statistics"]["true_source"]
    physical = values * float(stats["scale"]) + float(stats["mean"])
    if valid_mask is not None:
        if tuple(valid_mask.shape) != tuple(values.shape):
            raise ValueError("valid_mask shape must exactly match source values")
        physical = physical * valid_mask
    return physical


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(policy: str) -> torch.device:
    if policy == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is unavailable")
    if policy == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_report() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version, "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda, "cuda_available": cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda else "CPU",
        "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def atomic_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(final.name + ".part")
    if temporary.exists():
        raise TrainingError(f"Partial checkpoint already exists: {temporary}")
    try:
        torch.save(payload, temporary)
        temporary.replace(final)
    except Exception:
        raise


def load_checkpoint(
    path: str | Path, *, expected_dataset: dict[str, str] | None = None
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as exc:
        raise TrainingError(f"Cannot load checkpoint: {path}") from exc
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise TrainingError("Incompatible checkpoint schema version")
    if expected_dataset:
        for key in ("dataset_manifest_hash", "dataset_hdf5_hash", "configuration_hash", "normalization_hash"):
            if checkpoint.get(key) != expected_dataset.get(key):
                raise TrainingError(f"Checkpoint dataset hash mismatch: {key}")
    return checkpoint


def _dataset_hashes(dataset: SyntheticTorchDataset) -> dict[str, str]:
    manifest = dataset.manifest
    return {
        "dataset_manifest_hash": manifest["manifest_content_sha256"],
        "dataset_hdf5_hash": manifest["dataset_sha256"],
        "configuration_hash": manifest["configuration_sha256"],
        "normalization_hash": manifest["normalization_sha256"],
    }


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint_payload(
    *, model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: Any, scaler: torch.amp.GradScaler,
    train_generator: torch.Generator,
    epoch: int, best_validation_loss: float, best_epoch: int,
    best_validation_metrics: dict[str, float],
    epochs_without_improvement: int, config: dict[str, Any],
    dataset_hashes: dict[str, str], train_ids: list[str], validation_ids: list[str],
    device: torch.device, mixed_precision: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_architecture": config["architecture"],
        "nonnegative_policy": config["nonnegative_policy"],
        "source_zero_normalized": float(
            model.source_zero_normalized.detach().cpu().item()
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler_state_dict": scaler.state_dict(),
        "train_loader_generator_state": train_generator.get_state(),
        "rng_state": _capture_rng_state(),
        "epoch": epoch, "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "early_stopping_state": {"epochs_without_improvement": epochs_without_improvement},
        "random_seeds": {"global": config["seed"]},
        **dataset_hashes,
        "train_sample_ids": train_ids, "validation_sample_ids": validation_ids,
        "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "mixed_precision": mixed_precision,
        "training_configuration": config,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _run_epoch(
    *, model: ResidualAttentionUNet, loader: DataLoader,
    device: torch.device, loss_weights: dict[str, Any],
    normalization: dict[str, Any], optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None, mixed_precision: bool,
    gradient_clip_norm: float,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    weights = validate_loss_weights(loss_weights)
    loss_totals = {
        "squared_error": 0.0, "absolute_error": 0.0, "valid_nodes": 0.0,
        "gradient_error": 0.0, "valid_edges": 0.0,
    }
    metric_totals: dict[str, dict[str, float]] = {
        "model": _empty_metric_totals()
    }
    if not training:
        metric_totals.update({
            "identity": _empty_metric_totals(),
            "smoothness": _empty_metric_totals(),
        })
    sample_order: list[str] = []
    for batch in loader:
        sample_ids = batch.get("sample_id", [])
        sample_order.extend(
            [sample_ids] if isinstance(sample_ids, str) else list(sample_ids)
        )
        inputs = batch["inputs"].to(device, non_blocking=True)
        smooth = batch["smoothness_normalized"].to(device, non_blocking=True)
        target = batch["target_normalized"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = torch.amp.autocast(
            "cuda", enabled=mixed_precision,
            dtype=torch.bfloat16 if mixed_precision else None,
        )
        with context:
            prediction = model(inputs, smooth, mask)
            loss, components = masked_loss(prediction, target, mask, loss_weights)
        if not torch.isfinite(loss):
            raise TrainingError("Non-finite loss encountered")
        if training:
            assert optimizer is not None
            if scaler is not None and mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise TrainingError("Non-finite gradient encountered")
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if not torch.isfinite(gradient_norm):
                raise TrainingError("Non-finite gradient norm encountered")
            if scaler is not None and mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        _accumulate_loss_totals(
            loss_totals, prediction.detach(), target.detach(), mask.detach()
        )
        physical_prediction = denormalize_source(
            prediction.detach(), normalization, mask
        )
        physical_target = batch["target_physical"].to(device) * mask
        _accumulate_metric_totals(
            metric_totals["model"], physical_prediction, physical_target, mask
        )
        if not training:
            for baseline_name in ("identity", "smoothness"):
                baseline = batch[f"{baseline_name}_physical"].to(device)
                _accumulate_metric_totals(
                    metric_totals[baseline_name], baseline, physical_target, mask
                )
    if loss_totals["valid_nodes"] == 0:
        raise TrainingError("Loader produced no batches")
    mse = loss_totals["squared_error"] / loss_totals["valid_nodes"]
    l1 = loss_totals["absolute_error"] / loss_totals["valid_nodes"]
    gradient = (
        loss_totals["gradient_error"] / loss_totals["valid_edges"]
        if loss_totals["valid_edges"] else 0.0
    )
    result: dict[str, Any] = {
        "loss_mse": mse if weights["mse"] else 0.0,
        "loss_l1": l1 if weights["l1"] else 0.0,
        "loss_gradient": gradient if weights["gradient"] else 0.0,
        "sample_order": sample_order,
    }
    result["loss_total"] = (
        weights["mse"] * result["loss_mse"]
        + weights["l1"] * result["loss_l1"]
        + weights["gradient"] * result["loss_gradient"]
    )
    result.update(_finalize_metric_totals(metric_totals["model"]))
    if not training:
        for baseline_name in ("identity", "smoothness"):
            for key, value in _finalize_metric_totals(
                metric_totals[baseline_name]
            ).items():
                result[f"{baseline_name}_{key}"] = value
    return result


def _empty_metric_totals() -> dict[str, float]:
    return {
        "squared_error": 0.0, "absolute_error": 0.0,
        "valid_nodes": 0.0, "target_squared": 0.0,
        "maximum_absolute": 0.0,
    }


def _accumulate_metric_totals(
    totals: dict[str, float], prediction: Tensor, target: Tensor, mask: Tensor
) -> None:
    difference = (prediction - target) * mask
    totals["squared_error"] += float(difference.square().sum(dtype=torch.float64).item())
    totals["absolute_error"] += float(difference.abs().sum(dtype=torch.float64).item())
    totals["valid_nodes"] += float(mask.sum(dtype=torch.float64).item())
    totals["target_squared"] += float(
        ((target * mask).square()).sum(dtype=torch.float64).item()
    )
    totals["maximum_absolute"] = max(
        totals["maximum_absolute"], float(difference.abs().amax().item())
    )


def _finalize_metric_totals(totals: dict[str, float]) -> dict[str, float]:
    count = max(totals["valid_nodes"], 1.0)
    target_squared = max(totals["target_squared"], np.finfo(float).eps)
    return {
        "rmse": math.sqrt(totals["squared_error"] / count),
        "mae": totals["absolute_error"] / count,
        "relative_l2": math.sqrt(totals["squared_error"] / target_squared),
        "maximum_absolute": totals["maximum_absolute"],
    }


def _accumulate_loss_totals(
    totals: dict[str, float], prediction: Tensor, target: Tensor, mask: Tensor
) -> None:
    difference = prediction - target
    totals["squared_error"] += float(
        (difference.square() * mask).sum(dtype=torch.float64).item()
    )
    totals["absolute_error"] += float(
        (difference.abs() * mask).sum(dtype=torch.float64).item()
    )
    totals["valid_nodes"] += float(mask.sum(dtype=torch.float64).item())
    dx_mask = mask[..., 1:, :] * mask[..., :-1, :]
    dy_mask = mask[..., :, 1:] * mask[..., :, :-1]
    dx = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    dy = (prediction[..., :, 1:] - prediction[..., :, :-1]) - (
        target[..., :, 1:] - target[..., :, :-1]
    )
    totals["gradient_error"] += float(
        ((dx.abs() * dx_mask).sum(dtype=torch.float64)
        + (dy.abs() * dy_mask).sum(dtype=torch.float64)).item()
    )
    totals["valid_edges"] += float(
        (dx_mask.sum(dtype=torch.float64) + dy_mask.sum(dtype=torch.float64)).item()
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_state_sha256(state_dict: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Train on train/validation only and publish reproducible artifacts."""
    validate_model_config(config)
    require_supported_torch()
    set_determinism(config["seed"])
    device = select_device(config["device_policy"])
    mixed_precision = bool(config["mixed_precision"] and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    selected_train_ids = None
    selected_validation_ids = None
    selection_path = config.get("sample_selection_manifest")
    if selection_path:
        try:
            selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelConfigurationError("Cannot read sample selection manifest") from exc
        selected_train_ids = selection.get("train_sample_ids")
        selected_validation_ids = selection.get("validation_select_sample_ids")
        if not isinstance(selected_train_ids, list) or not isinstance(selected_validation_ids, list):
            raise ModelConfigurationError("Selection manifest must contain train_sample_ids and validation_select_sample_ids")
    train_loader, validation_loader, train_dataset, validation_dataset = make_loaders(
        config["dataset_directory"], batch_size=config["batch_size"],
        workers=config["workers"], seed=config["seed"],
        pin_memory=device.type == "cuda",
        train_sample_ids=selected_train_ids,
        validation_sample_ids=selected_validation_ids,
    )
    output = Path(config["output_directory"])
    checkpoints = Path(config["checkpoint_directory"])
    logs = Path(config["log_directory"])
    for directory in (output, checkpoints, logs):
        directory.mkdir(parents=True, exist_ok=True)
    _write_json(logs / "run_configuration.json", config)
    _write_json(logs / "environment.json", environment_report())
    _write_json(logs / "sample_ids.json", {
        "train": train_dataset.sample_ids, "validation": validation_dataset.sample_ids,
        "excluded_roles": ["test_id", "test_ood_shape", "test_ood_sensor", "test_ood_noise"],
    })
    model = build_model(config, train_dataset.normalization).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = None
    if config["scheduler"]["name"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"],
            eta_min=float(config["scheduler"].get("minimum_learning_rate", 0.0)),
        )
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    train_generator = train_loader.task3_generator  # type: ignore[attr-defined]
    dataset_hashes = _dataset_hashes(train_dataset)
    start_epoch = 1
    best_loss = math.inf
    best_epoch = 0
    best_validation_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    if config["resume"]:
        checkpoint = load_checkpoint(config["resume"], expected_dataset=dataset_hashes)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_metrics = dict(checkpoint["best_validation_metrics"])
        epochs_without_improvement = int(checkpoint["early_stopping_state"]["epochs_without_improvement"])
        train_generator.set_state(checkpoint["train_loader_generator_state"])
        _restore_rng_state(checkpoint["rng_state"])
        target_best = checkpoints / "best.pt"
        if not target_best.exists():
            source_best = Path(config["resume"]).resolve().parent / "best.pt"
            if source_best.is_file():
                preserved_best = load_checkpoint(
                    source_best, expected_dataset=dataset_hashes
                )
                atomic_checkpoint(target_best, preserved_best)
            elif int(checkpoint["epoch"]) == best_epoch:
                atomic_checkpoint(target_best, checkpoint)
    history_path = logs / "history.jsonl"
    if not config["resume"] and history_path.exists():
        history_path.unlink()
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            train_metrics = _run_epoch(
                model=model, loader=train_loader, device=device,
                loss_weights=config["loss_weights"], normalization=train_dataset.normalization,
                optimizer=optimizer, scaler=scaler, mixed_precision=mixed_precision,
                gradient_clip_norm=float(config["gradient_clip_norm"]),
            )
            with torch.no_grad():
                validation_metrics = _run_epoch(
                    model=model, loader=validation_loader, device=device,
                    loss_weights=config["loss_weights"], normalization=train_dataset.normalization,
                    optimizer=None, scaler=None, mixed_precision=mixed_precision,
                    gradient_clip_norm=float(config["gradient_clip_norm"]),
                )
            validation_loss = validation_metrics["loss_total"]
            improved = validation_loss < best_loss - float(config["early_stopping"]["minimum_improvement"])
            if improved:
                best_loss = validation_loss
                best_epoch = epoch
                best_validation_metrics = dict(validation_metrics)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if scheduler is not None:
                scheduler.step()
            record = {
                "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics, "validation": validation_metrics,
                "functional_only": bool(config["functional_only"]),
            }
            history.append(record)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            payload = _checkpoint_payload(
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, train_generator=train_generator, epoch=epoch,
                best_validation_loss=best_loss,
                best_epoch=best_epoch,
                best_validation_metrics=best_validation_metrics,
                epochs_without_improvement=epochs_without_improvement,
                config=config, dataset_hashes=dataset_hashes,
                train_ids=train_dataset.sample_ids,
                validation_ids=validation_dataset.sample_ids,
                device=device, mixed_precision=mixed_precision,
            )
            atomic_checkpoint(checkpoints / "last.pt", payload)
            if improved:
                atomic_checkpoint(checkpoints / "best.pt", payload)
            if epochs_without_improvement >= int(config["early_stopping"]["patience"]):
                break
    finally:
        train_dataset.close()
        validation_dataset.close()
    runtime = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    best_checkpoint = load_checkpoint(
        checkpoints / "best.pt", expected_dataset=dataset_hashes
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    _create_training_outputs(
        history, output, model, validation_dataset, device,
        train_dataset.normalization, config, best_epoch,
    )
    summary = {
        "run_label": config["run_label"], "device": str(device),
        "parameter_count": parameter_count(model), "epochs_completed": len(history),
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "best_validation_metrics": best_validation_metrics,
        "last_epoch": history[-1]["epoch"] if history else start_epoch - 1,
        "runtime_seconds": runtime, "peak_cuda_memory_bytes": peak_memory,
        "history_path": str(history_path.resolve()),
        "train_sample_ids": train_dataset.sample_ids,
        "validation_sample_ids": validation_dataset.sample_ids,
    }
    _write_json(logs / "summary.json", summary)
    return summary


def _create_training_outputs(
    history: list[dict[str, Any]], output: Path,
    model: ResidualAttentionUNet, validation_dataset: SyntheticTorchDataset,
    device: torch.device, normalization: dict[str, Any], config: dict[str, Any],
    best_epoch: int,
) -> None:
    if not history:
        return
    output.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot(epochs, [record["train"]["loss_total"] for record in history], label="train")
    axis.plot(epochs, [record["validation"]["loss_total"] for record in history], label="validation")
    axis.set(xlabel="epoch", ylabel="interior masked supervised loss", title="Functional smoke run only\nNot a scientific performance result")
    axis.legend()
    figure.savefig(output / "loss_curve.png", dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot(epochs, [record["validation"]["rmse"] for record in history], label="model RMSE")
    axis.plot(epochs, [record["validation"]["identity_rmse"] for record in history], label="identity RMSE")
    axis.plot(epochs, [record["validation"]["smoothness_rmse"] for record in history], label="smoothness RMSE")
    axis.set(xlabel="epoch", ylabel="physical interior RMSE", title="Functional smoke run only\nNot a scientific performance result")
    axis.legend()
    figure.savefig(output / "validation_metrics.png", dpi=120)
    plt.close(figure)
    sample = validation_dataset[0]
    model.eval()
    with torch.no_grad():
        prediction_normalized = model(
            sample["inputs"][None].to(device),
            sample["smoothness_normalized"][None].to(device),
            sample["valid_mask"][None].to(device),
        )
    prediction = denormalize_source(
        prediction_normalized.cpu(), normalization,
        sample["valid_mask"][None],
    )[0, 0].numpy()
    arrays = [
        (sample["target_physical"][0].numpy(), "True interior source q"),
        (sample["identity_physical"][0].numpy(), "Identity Tikhonov"),
        (sample["smoothness_physical"][0].numpy(), "Smoothness Tikhonov"),
        (prediction, "Residual-attention prediction"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for axis, (values, title) in zip(axes, arrays):
        image = axis.imshow(values, origin="lower")
        axis.set_title(title)
        axis.set_xlabel("grid j")
        axis.set_ylabel("grid i")
        figure.colorbar(image, ax=axis, shrink=0.72)
    figure.suptitle("Functional smoke run only - Not a scientific performance result")
    figure.savefig(output / "validation_prediction.png", dpi=120)
    plt.close(figure)
    _write_json(output / "validation_prediction.json", {
        "sample_id": sample["sample_id"], "split": sample["split"],
        "checkpoint": "best.pt", "best_epoch": best_epoch,
        "model_state_sha256": model_state_sha256(model.state_dict()),
        "label": "Functional smoke run only - Not a scientific performance result",
        "normalization": "target and source baselines use persisted true_source statistics",
    })
    validation_dataset.close()


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(path)
    return {
        key: checkpoint[key] for key in (
            "checkpoint_schema_version", "model_architecture", "nonnegative_policy",
            "epoch", "best_epoch", "best_validation_loss",
            "best_validation_metrics", "random_seeds", "dataset_manifest_hash",
            "dataset_hdf5_hash", "configuration_hash", "normalization_hash",
            "train_sample_ids", "validation_sample_ids", "torch_version", "cuda_build",
            "device_name", "mixed_precision", "git_commit", "timestamp_utc",
        )
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect-environment")
    for name in ("validate-config", "model-summary", "train"):
        subparser = commands.add_parser(name)
        subparser.add_argument("config", type=Path)
    resume = commands.add_parser("resume")
    resume.add_argument("config", type=Path)
    resume.add_argument("checkpoint", type=Path)
    inspect_checkpoint_parser = commands.add_parser("inspect-checkpoint")
    inspect_checkpoint_parser.add_argument("checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect-environment":
            report = environment_report()
        elif args.command == "inspect-checkpoint":
            report = checkpoint_summary(args.checkpoint)
        else:
            config = load_model_config(args.config)
            if args.command == "validate-config":
                report = {"valid": True, "schema_version": config["schema_version"], "functional_only": config["functional_only"]}
            elif args.command == "model-summary":
                model = build_model(config)
                size = 8 if config["functional_only"] else 32
                with torch.no_grad():
                    output = model(torch.zeros(1, 4, size, size), torch.zeros(1, 1, size, size), torch.ones(1, 1, size, size))
                report = {"parameter_count": parameter_count(model), "output_shape": list(output.shape), "architecture": config["architecture"]}
            else:
                if args.command == "resume":
                    config = json.loads(json.dumps(config))
                    config["resume"] = str(args.checkpoint)
                report = train(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ModelConfigurationError, DatasetPipelineError, TrainingError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
