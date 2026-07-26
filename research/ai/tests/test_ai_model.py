"""CPU-only, network-free tests for the isolated Task 3 training system."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "ai_model.py"
SPEC = importlib.util.spec_from_file_location("ai_model", MODULE_PATH)
ai_model = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ai_model
SPEC.loader.exec_module(ai_model)
ai_data = sys.modules["ai_data"]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("task3_data") / "dataset"
    config = {
        "schema_version": 2, "random_seed": 31, "output_directory": str(output),
        "grid_shape": [8, 8], "num_samples": 6,
        "source_family_probabilities": {"one_gaussian": 1.0, "sharp_edged": 0.0},
        "source_count_range": [2, 2], "allow_signed_sources": False,
        "signed_probability": 0.0, "amplitude_range": [1.0, 1.2],
        "width_range": [0.1, 0.12], "size_range": [0.1, 0.2],
        "sensor_strategies": ["regular_grid", "random", "center_focused"],
        "sensor_count_range": [6, 7], "sensor_seeds": [41],
        "noise_levels": [0.0, 0.01, 0.1],
        "identity_alpha_choices": [0.01], "smoothness_alpha_choices": [0.001],
        "split_rules": {"counts": {role: 1 for role in ai_data.SPLIT_ROLES}},
        "ood_source_families": ["sharp_edged"],
        "ood_sensor_strategies": ["center_focused"], "ood_noise_levels": [0.1],
        "normalization": {"enabled": True, "method": "global_standard"},
        "storage_compression": "gzip", "preview_count": 1,
    }
    ai_data.build_synthetic_dataset(config, output_directory=output)
    return output


def model_config(dataset: Path, root: Path, *, epochs: int = 1) -> dict:
    return {
        "schema_version": 1, "run_label": "Functional smoke run only - Not a scientific performance result",
        "seed": 7, "dataset_directory": str(dataset),
        "output_directory": str(root / "outputs"),
        "checkpoint_directory": str(root / "checkpoints"),
        "log_directory": str(root / "logs"), "device_policy": "cpu",
        "architecture": {"input_channels": 4, "output_channels": 1, "base_channels": 2, "depth": 2, "dropout": 0.0, "upsampling": "bilinear"},
        "nonnegative_policy": "softplus", "batch_size": 1, "workers": 0,
        "epochs": epochs, "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0},
        "scheduler": {"name": "none"}, "loss_weights": {"mse": 1.0, "l1": 0.1, "gradient": 0.05},
        "mixed_precision": True, "gradient_clip_norm": 1.0,
        "early_stopping": {"patience": 3, "minimum_improvement": 0.0},
        "history_format": "jsonl", "resume": None, "torch_compile": False,
        "functional_only": True,
    }


@pytest.fixture(scope="module")
def trained_run(tiny_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict, Path]:
    root = tmp_path_factory.mktemp("task3_train")
    config = model_config(tiny_dataset, root)
    summary = ai_model.train(config)
    return config, summary, root


def test_torch_version_and_environment() -> None:
    ai_model.require_supported_torch()
    assert ai_model._version_tuple(torch.__version__) >= (2, 1)
    report = ai_model.environment_report()
    assert report["torch_version"] == torch.__version__


def test_configuration_validation_and_invalid_architecture(tiny_dataset: Path, tmp_path: Path) -> None:
    config = model_config(tiny_dataset, tmp_path)
    assert ai_model.validate_model_config(config) is config
    invalid = json.loads(json.dumps(config))
    invalid["architecture"]["input_channels"] = 3
    with pytest.raises(ai_model.ModelConfigurationError, match="four inputs"):
        ai_model.validate_model_config(invalid)
    invalid = json.loads(json.dumps(config))
    invalid["torch_compile"] = True
    with pytest.raises(ai_model.ModelConfigurationError, match="disabled"):
        ai_model.validate_model_config(invalid)


@pytest.mark.parametrize("size", [8, 32])
def test_forward_shape_compatibility_and_boundaries(size: int) -> None:
    model = ai_model.ResidualAttentionUNet(base_channels=4, depth=2, dropout=0, nonnegative_policy="none")
    inputs = torch.randn(2, 4, size, size)
    smooth = torch.randn(2, 1, size, size)
    mask = torch.zeros(2, 1, size, size)
    mask[..., 1:-1, 1:-1] = 1
    output = model(inputs, smooth, mask)
    assert output.shape == (2, 1, size, size)
    assert torch.count_nonzero(output[..., 0, :]) == 0
    assert torch.count_nonzero(output[..., -1, :]) == 0
    assert torch.count_nonzero(output[..., :, 0]) == 0
    assert torch.count_nonzero(output[..., :, -1]) == 0


def test_incompatible_spatial_size_rejected() -> None:
    model = ai_model.ResidualAttentionUNet(base_channels=2, depth=3)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.zeros(1, 4, 10, 10), torch.zeros(1, 1, 10, 10), torch.ones(1, 1, 10, 10))


def test_forward_rejects_incompatible_baseline_and_mask_shapes() -> None:
    model = ai_model.ResidualAttentionUNet(base_channels=2, depth=1)
    inputs = torch.zeros(2, 4, 8, 8)
    with pytest.raises(ValueError, match="smoothness_normalized"):
        model(inputs, torch.zeros(1, 1, 8, 8), torch.ones(2, 1, 8, 8))
    with pytest.raises(ValueError, match="valid_mask"):
        model(inputs, torch.zeros(2, 1, 8, 8), torch.ones(2, 8, 8))


def test_parameter_ceiling_default_and_deterministic_initialization(tiny_dataset: Path, tmp_path: Path) -> None:
    config = model_config(tiny_dataset, tmp_path)
    config["architecture"].update({"base_channels": 32, "depth": 3})
    first = ai_model.build_model(config)
    second = ai_model.build_model(config)
    assert ai_model.parameter_count(first) <= 5_000_000
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))


@pytest.mark.parametrize("policy", ["none", "relu", "softplus"])
def test_residual_addition_mask_and_nonnegative_policies(policy: str) -> None:
    model = ai_model.ResidualAttentionUNet(base_channels=2, depth=1, dropout=0, nonnegative_policy=policy)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    smooth = torch.full((1, 1, 8, 8), -1.0)
    mask = torch.zeros_like(smooth)
    mask[..., 1:-1, 1:-1] = 1
    output = model(torch.zeros(1, 4, 8, 8), smooth, mask)
    if policy == "none":
        assert torch.all(output[..., 1:-1, 1:-1] == -1)
    else:
        assert torch.all(output >= 0)
    assert torch.count_nonzero(output * (1 - mask)) == 0


@pytest.mark.parametrize("policy", ["none", "relu", "softplus"])
def test_physical_zero_threshold_policies_and_exact_boundaries(policy: str) -> None:
    normalization = {"statistics": {"true_source": {"mean": 2.0, "scale": 4.0}}}
    threshold = ai_model.source_zero_from_normalization(normalization)
    model = ai_model.ResidualAttentionUNet(
        base_channels=2, depth=1, dropout=0,
        nonnegative_policy=policy, source_zero_normalized=threshold,
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    smooth = torch.full((1, 1, 8, 8), -1.0)
    mask = torch.zeros_like(smooth); mask[..., 1:-1, 1:-1] = 1
    normalized = model(torch.zeros(1, 4, 8, 8), smooth, mask)
    assert torch.all(normalized[..., 0, :] == threshold)
    physical = ai_model.denormalize_source(normalized, normalization, mask)
    assert torch.all(physical[..., 0, :] == 0.0)
    assert torch.all(physical[..., -1, :] == 0.0)
    if policy == "none":
        assert torch.any(physical[..., 1:-1, 1:-1] < 0)
    else:
        assert torch.all(physical >= 0)


def test_dataset_split_filtering_normalization_channels_and_no_mutation(tiny_dataset: Path) -> None:
    before = file_digest(tiny_dataset / "synthetic_dataset.h5")
    train = ai_model.SyntheticTorchDataset(tiny_dataset, "train")
    validation = ai_model.SyntheticTorchDataset(tiny_dataset, "validation")
    assert len(train) == len(validation) == 1
    assert not set(train.sample_ids) & set(validation.sample_ids)
    assert all(item["split"] == "train" for item in [train[0]])
    sample = train[0]
    assert sample["inputs"].shape == (4, 8, 8)
    assert set(torch.unique(sample["inputs"][1]).tolist()) <= {0.0, 1.0}
    stats = train.normalization["statistics"]["true_source"]
    expected = (sample["identity_physical"] - stats["mean"]) / stats["scale"]
    assert torch.allclose(sample["inputs"][2], expected[0])
    assert train.normalization["sample_ids"] == train.sample_ids
    assert file_digest(tiny_dataset / "synthetic_dataset.h5") == before
    train.close()
    validation.close()
    assert train._reader is None and validation._reader is None


def test_external_and_empty_splits_rejected(tiny_dataset: Path) -> None:
    with pytest.raises(ValueError, match="train or validation"):
        ai_model.SyntheticTorchDataset(tiny_dataset, "test_id")


def test_empty_validation_split_rejected(tiny_dataset: Path, tmp_path: Path) -> None:
    copied = tmp_path / "empty_validation"
    shutil.copytree(tiny_dataset, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for sample in manifest["samples"]:
        if sample["split"] == "validation":
            sample["split"] = "train"
    manifest["manifest_content_sha256"] = ai_data._manifest_content_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ai_data.DatasetPipelineError, match="empty"):
        ai_model.SyntheticTorchDataset(copied, "validation")


def test_lazy_reader_and_context_close(tiny_dataset: Path) -> None:
    dataset = ai_model.SyntheticTorchDataset(tiny_dataset, "train")
    assert dataset._reader is None
    with dataset:
        dataset[0]
        assert dataset._reader is not None
    assert dataset._reader is None


def test_deterministic_train_shuffle_and_validation_sampler(tiny_dataset: Path) -> None:
    first = ai_model.make_loaders(tiny_dataset, batch_size=1, workers=0, seed=9, pin_memory=False)
    second = ai_model.make_loaders(tiny_dataset, batch_size=1, workers=0, seed=9, pin_memory=False)
    assert isinstance(first[0].sampler, torch.utils.data.RandomSampler)
    assert isinstance(first[1].sampler, torch.utils.data.SequentialSampler)
    assert [batch["sample_id"][0] for batch in first[0]] == [batch["sample_id"][0] for batch in second[0]]
    for collection in (first, second):
        collection[2].close(); collection[3].close()


def test_masked_mse_l1_gradient_and_boundary_exclusion() -> None:
    target = torch.zeros(1, 1, 4, 4)
    prediction = torch.zeros_like(target)
    prediction[..., 1:3, 1:3] = 2
    prediction[..., 0, :] = 1000
    mask = torch.zeros_like(target); mask[..., 1:3, 1:3] = 1
    total, parts = ai_model.masked_loss(prediction, target, mask, {"mse": 1, "l1": 1, "gradient": 1})
    assert parts["mse"].item() == 4
    assert parts["l1"].item() == 2
    assert parts["gradient"].item() == 0
    assert total.item() == 6


@pytest.mark.parametrize("weights", [
    {"mse": -1, "l1": 0, "gradient": 0},
    {"mse": 0, "l1": 0, "gradient": 0},
])
def test_invalid_loss_weights(weights: dict) -> None:
    with pytest.raises(ai_model.ModelConfigurationError):
        ai_model.validate_loss_weights(weights)


def test_gradient_loss_detects_interior_differences() -> None:
    prediction = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction)
    _, parts = ai_model.masked_loss(prediction, target, mask, {"mse": 0, "l1": 0, "gradient": 1})
    assert parts["gradient"] > 0


def _metric_batch(values: list[float], ids: list[str]) -> dict:
    batch = len(values)
    target = torch.stack([
        torch.full((1, 4, 4), value, dtype=torch.float32) for value in values
    ])
    smooth = torch.stack([
        torch.full((1, 4, 4), value + 0.5, dtype=torch.float32) for value in values
    ])
    identity = torch.stack([
        torch.full((1, 4, 4), value - 0.25, dtype=torch.float32) for value in values
    ])
    mask = torch.ones(batch, 1, 4, 4)
    return {
        "inputs": torch.zeros(batch, 4, 4, 4),
        "smoothness_normalized": smooth.clone(),
        "target_normalized": target.clone(), "valid_mask": mask,
        "target_physical": target, "identity_physical": identity,
        "smoothness_physical": smooth, "sample_id": ids,
    }


def test_epoch_and_baseline_metrics_are_batch_partition_invariant() -> None:
    class BaselineModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor, smooth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return smooth

    common = {
        "model": BaselineModel(), "device": torch.device("cpu"),
        "loss_weights": {"mse": 1, "l1": 0.1, "gradient": 0.05},
        "normalization": {"statistics": {"true_source": {"mean": 0, "scale": 1}}},
        "optimizer": None, "scaler": None, "mixed_precision": False,
        "gradient_clip_norm": 1.0,
    }
    grouped = ai_model._run_epoch(
        loader=[_metric_batch([1.0, 3.0], ["a", "b"])], **common
    )
    separated = ai_model._run_epoch(
        loader=[_metric_batch([1.0], ["a"]), _metric_batch([3.0], ["b"])],
        **common,
    )
    for key in grouped:
        if key != "sample_order":
            assert grouped[key] == pytest.approx(separated[key], abs=1e-12)
    assert grouped["sample_order"] == separated["sample_order"] == ["a", "b"]


def test_optimizer_step_clipping_and_nonfinite_loss() -> None:
    model = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss = model(torch.ones(1, 1, 2, 2)).square().mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    assert torch.isfinite(norm)
    optimizer.step()
    assert not torch.equal(before, model.weight)
    nan_loss, _ = ai_model.masked_loss(torch.tensor([[[[float("nan")]]]]), torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1), {"mse": 1, "l1": 0, "gradient": 0})
    assert not torch.isfinite(nan_loss)


def test_training_epoch_rejects_nonfinite_model_output() -> None:
    class NanModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor, smooth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return inputs[:, :1] * float("nan")

    batch = {
        "inputs": torch.zeros(1, 4, 4, 4),
        "smoothness_normalized": torch.zeros(1, 1, 4, 4),
        "target_normalized": torch.zeros(1, 1, 4, 4),
        "valid_mask": torch.ones(1, 1, 4, 4),
        "target_physical": torch.zeros(1, 1, 4, 4),
        "identity_physical": torch.zeros(1, 1, 4, 4),
        "smoothness_physical": torch.zeros(1, 1, 4, 4),
    }
    with pytest.raises(ai_model.TrainingError, match="Non-finite loss"):
        ai_model._run_epoch(
            model=NanModel(), loader=[batch], device=torch.device("cpu"),
            loss_weights={"mse": 1, "l1": 0, "gradient": 0},
            normalization={"statistics": {"true_source": {"mean": 0, "scale": 1}}},
            optimizer=None, scaler=None, mixed_precision=False, gradient_clip_norm=1,
        )


def test_one_epoch_training_metrics_outputs_and_cpu_amp_fallback(trained_run: tuple[dict, dict, Path]) -> None:
    config, summary, root = trained_run
    assert summary["epochs_completed"] == 1
    assert np.isfinite(summary["best_validation_loss"])
    assert summary["peak_cuda_memory_bytes"] == 0
    history = [json.loads(line) for line in (root / "logs" / "history.jsonl").read_text().splitlines()]
    validation = history[0]["validation"]
    for key in ("rmse", "mae", "relative_l2", "maximum_absolute", "identity_rmse", "smoothness_rmse"):
        assert np.isfinite(validation[key])
    assert (root / "outputs" / "validation_prediction.png").is_file()
    assert config["mixed_precision"] is True


def test_checkpoints_atomic_schema_provenance_and_history(trained_run: tuple[dict, dict, Path]) -> None:
    _, summary, root = trained_run
    for name in ("best.pt", "last.pt"):
        path = root / "checkpoints" / name
        assert path.is_file() and not path.with_name(path.name + ".part").exists()
        checkpoint = ai_model.load_checkpoint(path)
        assert checkpoint["checkpoint_schema_version"] == 2
        assert checkpoint["train_sample_ids"] == summary["train_sample_ids"]
        assert checkpoint["validation_sample_ids"] == summary["validation_sample_ids"]
        assert checkpoint["mixed_precision"] is False
    assert (root / "logs" / "history.jsonl").is_file()
    ids = json.loads((root / "logs" / "sample_ids.json").read_text())
    assert ids["train"] == summary["train_sample_ids"]
    assert ids["validation"] == summary["validation_sample_ids"]


def test_checkpoint_schema_and_dataset_hash_mismatch(trained_run: tuple[dict, dict, Path], tmp_path: Path) -> None:
    _, _, root = trained_run
    checkpoint = ai_model.load_checkpoint(root / "checkpoints" / "last.pt")
    bad = dict(checkpoint); bad["checkpoint_schema_version"] = 99
    bad_path = tmp_path / "bad.pt"; torch.save(bad, bad_path)
    with pytest.raises(ai_model.TrainingError, match="schema"):
        ai_model.load_checkpoint(bad_path)
    with pytest.raises(ai_model.TrainingError, match="hash mismatch"):
        ai_model.load_checkpoint(root / "checkpoints" / "last.pt", expected_dataset={
            "dataset_manifest_hash": "bad", "dataset_hdf5_hash": checkpoint["dataset_hdf5_hash"],
            "configuration_hash": checkpoint["configuration_hash"], "normalization_hash": checkpoint["normalization_hash"],
        })


def _assert_nested_equal(first: object, second: object) -> None:
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor) and torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert isinstance(second, np.ndarray) and np.array_equal(first, second)
    elif isinstance(first, dict):
        assert isinstance(second, dict) and first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert isinstance(second, type(first)) and len(first) == len(second)
        for left, right in zip(first, second):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_resume_is_equivalent_to_uninterrupted_cpu_training(
    tiny_dataset: Path, trained_run: tuple[dict, dict, Path], tmp_path: Path
) -> None:
    _, _, trained_root = trained_run
    config = model_config(tiny_dataset, tmp_path / "resume", epochs=2)
    config["resume"] = str(trained_root / "checkpoints" / "last.pt")
    resumed_summary = ai_model.train(config)
    uninterrupted_config = model_config(
        tiny_dataset, tmp_path / "uninterrupted", epochs=2
    )
    uninterrupted_summary = ai_model.train(uninterrupted_config)
    resumed = ai_model.load_checkpoint(
        Path(config["checkpoint_directory"]) / "last.pt"
    )
    uninterrupted = ai_model.load_checkpoint(
        Path(uninterrupted_config["checkpoint_directory"]) / "last.pt"
    )
    for key in (
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "grad_scaler_state_dict", "train_loader_generator_state", "rng_state",
    ):
        _assert_nested_equal(resumed[key], uninterrupted[key])
    resumed_history = [
        json.loads(line) for line in Path(resumed_summary["history_path"]).read_text().splitlines()
    ]
    uninterrupted_history = [
        json.loads(line) for line in Path(uninterrupted_summary["history_path"]).read_text().splitlines()
    ]
    assert resumed_history[0]["epoch"] == 2
    assert resumed_history[0]["train"] == uninterrupted_history[1]["train"]
    assert resumed_history[0]["validation"] == uninterrupted_history[1]["validation"]
    assert resumed["best_epoch"] == uninterrupted["best_epoch"]


def test_best_checkpoint_drives_qualitative_output(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    config = model_config(tiny_dataset, tmp_path / "best_output", epochs=2)
    config["early_stopping"]["minimum_improvement"] = 1e9
    summary = ai_model.train(config)
    best = ai_model.load_checkpoint(Path(config["checkpoint_directory"]) / "best.pt")
    last = ai_model.load_checkpoint(Path(config["checkpoint_directory"]) / "last.pt")
    metadata = json.loads(
        (Path(config["output_directory"]) / "validation_prediction.json").read_text()
    )
    assert summary["best_epoch"] == best["best_epoch"] == 1
    assert last["epoch"] == 2 and last["best_epoch"] == 1
    assert metadata["checkpoint"] == "best.pt" and metadata["best_epoch"] == 1
    assert metadata["model_state_sha256"] == ai_model.model_state_sha256(
        best["model_state_dict"]
    )
    assert metadata["model_state_sha256"] != ai_model.model_state_sha256(
        last["model_state_dict"]
    )


def test_training_never_reads_test_ids(tiny_dataset: Path, trained_run: tuple[dict, dict, Path]) -> None:
    _, summary, _ = trained_run
    manifest = json.loads((tiny_dataset / "manifest.json").read_text())
    test_ids = {sample["sample_id"] for sample in manifest["samples"] if sample["split"].startswith("test")}
    assert not test_ids & set(summary["train_sample_ids"])
    assert not test_ids & set(summary["validation_sample_ids"])


def test_no_network_or_external_paths_in_module() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "urllib" not in source and "requests" not in source
    assert "ThermoReconLab_external" not in source
