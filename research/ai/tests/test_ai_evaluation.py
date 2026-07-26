"""Fast CPU-only tests for Task 4 synthetic evaluation."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "ai_evaluation.py"
SPEC = importlib.util.spec_from_file_location("ai_evaluation", MODULE_PATH)
ai_evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ai_evaluation
SPEC.loader.exec_module(ai_evaluation)
ai_data = sys.modules["ai_data"]
ai_model = sys.modules["ai_model"]


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("task4") / "dataset"
    config = {
        "schema_version": 2, "random_seed": 77, "output_directory": str(output),
        "grid_shape": [8, 8], "num_samples": 12,
        "source_family_probabilities": {"one_gaussian": 1.0, "sharp_edged": 0.0},
        "source_count_range": [2, 2], "allow_signed_sources": False,
        "signed_probability": 0.0, "amplitude_range": [1.0, 1.2],
        "width_range": [0.1, 0.12], "size_range": [0.1, 0.2],
        "sensor_strategies": ["regular_grid", "random", "center_focused"],
        "sensor_count_range": [6, 7], "sensor_seeds": [41],
        "noise_levels": [0.0, 0.01, 0.1], "identity_alpha_choices": [0.01],
        "smoothness_alpha_choices": [0.001],
        "split_rules": {"counts": {"train": 2, "validation": 2, "test_id": 2, "test_ood_shape": 2, "test_ood_sensor": 2, "test_ood_noise": 2}},
        "ood_source_families": ["sharp_edged"], "ood_sensor_strategies": ["center_focused"],
        "ood_noise_levels": [0.1], "normalization": {"enabled": True, "method": "global_standard"},
        "storage_compression": "gzip", "preview_count": 1,
    }
    ai_data.build_synthetic_dataset(config, output_directory=output)
    return output


def evaluation_config(dataset: Path) -> dict:
    return {
        "schema_version": 1, "seed": 1, "partition_seed": 2,
        "dataset_directory": str(dataset), "base_model_configuration": "unused.json",
        "validation_select_count": 1, "validation_calibration_count": 1,
        "model_runs": [
            {"name": "full_residual_attention", "attention": True, "prediction_mode": "residual", "input_channel_mask": [1, 1, 1, 1], "epochs": 1, "batch_size": 2, "workers": 0, "device_policy": "cpu"},
            {"name": "residual_no_attention", "attention": False, "prediction_mode": "residual", "input_channel_mask": [1, 1, 1, 1], "epochs": 1, "batch_size": 2, "workers": 0, "device_policy": "cpu"},
            {"name": "direct_sparse_mask", "attention": True, "prediction_mode": "direct", "input_channel_mask": [1, 1, 0, 0], "epochs": 1, "batch_size": 2, "workers": 0, "device_policy": "cpu"},
        ],
        "checkpoint_directory": "c", "output_directory": "o", "log_directory": "l",
        "test_roles": list(ai_evaluation.TEST_ROLES), "bootstrap_repetitions": 20,
        "bootstrap_confidence_level": 0.9, "uncertainty_method": "mc_dropout",
        "mc_dropout_passes": 3, "uncertainty_std_floor": 1e-6,
        "target_interval_coverage": 0.9, "evaluation_batch_size": 2,
        "preview_sample_count": 1, "physics_evaluation_enabled": True,
        "external_manifest": None, "sensor_count_bins": [0, 10, 100],
    }


def test_configuration_validation(tiny_dataset: Path) -> None:
    config = evaluation_config(tiny_dataset)
    assert ai_evaluation.validate_evaluation_config(config) is config
    invalid = dict(config, uncertainty_method="posterior")
    with pytest.raises(ai_evaluation.EvaluationError):
        ai_evaluation.validate_evaluation_config(invalid)


@pytest.mark.parametrize("mutation", [
    lambda c: c["model_runs"][0].pop("attention"),
    lambda c: c["model_runs"][0].update(attention=False),
    lambda c: c["model_runs"][1].update(epochs=0),
    lambda c: c["model_runs"][2].update(batch_size=0),
    lambda c: c["model_runs"][2].update(workers=-1),
    lambda c: c.update(sensor_count_bins=[0, math.inf]),
    lambda c: c.update(test_roles=["test_id"] * 4),
    lambda c: c.update(external_manifest=object()),
    lambda c: c.update(physics_evaluation_enabled=False),
    lambda c: c.update(preview_sample_count=999),
])
def test_full_run_definition_and_configuration_rejection(tiny_dataset: Path, mutation) -> None:
    config = evaluation_config(tiny_dataset); mutation(config)
    with pytest.raises(ai_evaluation.EvaluationError): ai_evaluation.validate_evaluation_config(config)


def test_validation_partition_is_deterministic_disjoint_and_complete(tiny_dataset: Path) -> None:
    first = ai_evaluation.partition_validation(tiny_dataset, partition_seed=3, select_count=1, calibration_count=1)
    second = ai_evaluation.partition_validation(tiny_dataset, partition_seed=3, select_count=1, calibration_count=1)
    assert first == second
    select = set(first["validation_select_sample_ids"]); calibration = set(first["validation_calibration_sample_ids"])
    assert not select & calibration
    assert len(select | calibration) == 2
    assert not (set(first["train_sample_ids"]) & (select | calibration))


def test_explicit_task3_selection_and_invalid_ids(tiny_dataset: Path) -> None:
    partition = ai_evaluation.partition_validation(tiny_dataset, partition_seed=3, select_count=1, calibration_count=1)
    selected = ai_model.SyntheticTorchDataset(tiny_dataset, "validation", partition["validation_select_sample_ids"])
    assert selected.sample_ids == partition["validation_select_sample_ids"]
    selected.close()
    with pytest.raises(ai_data.DatasetPipelineError, match="Duplicate"):
        ai_model.SyntheticTorchDataset(tiny_dataset, "validation", [partition["validation_select_sample_ids"][0]] * 2)
    with pytest.raises(ai_data.DatasetPipelineError, match="Unknown"):
        ai_model.SyntheticTorchDataset(tiny_dataset, "validation", ["not-an-id"])


@pytest.fixture
def verification_case(tiny_dataset: Path, tmp_path: Path) -> tuple[dict, dict, dict]:
    config = evaluation_config(tiny_dataset)
    config.update(
        base_model_configuration=str(Path(__file__).parents[1] / "configs" / "model_smoke.json"),
        checkpoint_directory=str(tmp_path / "checkpoints"), log_directory=str(tmp_path / "logs"),
        output_directory=str(tmp_path / "outputs"),
    )
    partition = ai_evaluation.partition_validation(tiny_dataset, partition_seed=2, select_count=1, calibration_count=1)
    run_configs = ai_evaluation._compose_run_configs(config)
    hashes, _ = ai_evaluation.dataset_hashes_from_manifest(tiny_dataset)
    for name, run_config in run_configs.items():
        checkpoint = {
            "checkpoint_schema_version": ai_model.CHECKPOINT_SCHEMA_VERSION, **hashes,
            "train_sample_ids": list(partition["train_sample_ids"]),
            "validation_sample_ids": list(partition["validation_select_sample_ids"]),
            "epoch": 1, "best_epoch": 1, "model_architecture": deepcopy(run_config["architecture"]),
            "nonnegative_policy": run_config["nonnegative_policy"],
            "training_configuration": {"dataset_directory": str(tiny_dataset)},
        }
        path = Path(config["checkpoint_directory"]) / name / "best.pt"; path.parent.mkdir(parents=True); torch.save(checkpoint, path)
    return config, partition, run_configs


def _mutate_checkpoint(case, mutation) -> None:
    config, _, _ = case; path = Path(config["checkpoint_directory"]) / "full_residual_attention" / "best.pt"
    checkpoint = torch.load(path, weights_only=False); mutation(checkpoint); torch.save(checkpoint, path)


def test_checkpoint_verification_success_and_machine_readable_output(verification_case, tmp_path: Path) -> None:
    config, partition, run_configs = verification_case; output = tmp_path / "verification.json"
    result = ai_evaluation.verify_checkpoints(config, partition, run_configs, output_path=output)
    assert result["verified"] and result["partition"]["train_sample_count"] == 2
    assert json.loads(output.read_text())["dataset_hashes"] == result["dataset_hashes"]


@pytest.mark.parametrize("corruption", [
    "dataset_hash", "train_id", "validation_id", "calibration_leak", "test_leak", "architecture", "best_epoch",
])
def test_checkpoint_corruption_is_rejected(verification_case, corruption: str) -> None:
    config, partition, run_configs = verification_case
    reader = ai_data.SyntheticDatasetReader(config["dataset_directory"])
    test_id = next(sample["sample_id"] for sample in reader.manifest["samples"] if sample["split"] == "test_id"); reader.close()
    def mutate(checkpoint):
        if corruption == "dataset_hash": checkpoint["dataset_manifest_hash"] = "0" * 64
        elif corruption == "train_id": checkpoint["train_sample_ids"][0] = "wrong"
        elif corruption == "validation_id": checkpoint["validation_sample_ids"][0] = "wrong"
        elif corruption == "calibration_leak": checkpoint["validation_sample_ids"][0] = partition["validation_calibration_sample_ids"][0]
        elif corruption == "test_leak": checkpoint["train_sample_ids"][0] = test_id
        elif corruption == "architecture": checkpoint["model_architecture"]["attention"] = False
        elif corruption == "best_epoch": checkpoint["best_epoch"] = 2
    _mutate_checkpoint(verification_case, mutate)
    with pytest.raises(ai_evaluation.EvaluationError): ai_evaluation.verify_checkpoints(config, partition, run_configs)


def test_architecture_ablation_options_and_channel_mask() -> None:
    residual = ai_model.ResidualAttentionUNet(base_channels=2, depth=1, dropout=0, attention=False, prediction_mode="residual", input_channel_mask=[1, 1, 1, 1], nonnegative_policy="none")
    direct = ai_model.ResidualAttentionUNet(base_channels=2, depth=1, dropout=0, attention=True, prediction_mode="direct", input_channel_mask=[1, 1, 0, 0], nonnegative_policy="none")
    assert not any(isinstance(module, ai_model.ChannelSpatialAttention) for module in residual.modules())
    assert direct.prediction_mode == "direct"
    assert direct.input_channel_mask.flatten().tolist() == [1, 1, 0, 0]
    for model in (residual, direct):
        output = model(torch.zeros(1, 4, 8, 8), torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8))
        assert output.shape == (1, 1, 8, 8)


def test_source_metrics_peak_boundary_and_nonnegative() -> None:
    target = np.zeros((5, 5)); target[2, 2] = 2
    prediction = np.zeros((5, 5)); prediction[2, 3] = 1; prediction[1, 1] = -1; prediction[0, 0] = 0.5
    mask = np.zeros((5, 5), bool); mask[1:-1, 1:-1] = True
    result = ai_evaluation.source_metrics(prediction, target, mask)
    assert result["source_peak_location_distance"] == 1
    assert result["boundary_maximum_absolute"] == 0.5
    assert result["nonnegative_violation_fraction"] == pytest.approx(1 / 9)


def test_physics_metric_exact_source_is_zero(tiny_dataset: Path) -> None:
    reader = ai_data.SyntheticDatasetReader(tiny_dataset)
    sample = reader[0]; reader.close()
    metrics = ai_evaluation.physics_metrics(sample["true_source"], sample)
    assert metrics["physics_temperature_rmse"] < 1e-12
    assert metrics["clean_sensor_residual_rms"] < 1e-12


def test_aggregation_and_stratified_counts() -> None:
    rows = []
    for sample_id, error, valid_count in (("a", 1., 1), ("b", 3., 3)):
        row = {"sample_id": sample_id, "method": "m", "test_role": "r", "source_family": "f", "sensor_strategy": "s", "noise_level": 0.0, "sensor_count_bin": "b"}
        target = np.ones((3, 3)); prediction = target + error; mask = np.zeros((3, 3), bool); mask.flat[:valid_count] = True
        row.update(ai_evaluation.source_metrics(prediction, target, mask)); rows.append(row)
    result = ai_evaluation.aggregate_metrics(rows, ("method", "test_role"))
    mean = next(row for row in result if row["aggregation_type"] == "mean_per_sample")
    pooled = next(row for row in result if row["aggregation_type"] == "pooled_global")
    assert mean["sample_count"] == 2 and mean["rmse"] != pooled["rmse"]
    assert pooled["rmse"] == pytest.approx(math.sqrt(sum(row["source_squared_error_sum"] for row in rows) / 4))
    assert pooled["relative_l2"] == pytest.approx(math.sqrt(sum(row["source_squared_error_sum"] for row in rows) / sum(row["source_target_squared_sum"] for row in rows)))
    assert pooled["maximum_absolute_error"] == max(row["maximum_absolute_error"] for row in rows)
    reverse = ai_evaluation.aggregate_metrics(list(reversed(rows)), ("method", "test_role"))
    assert next(row for row in reverse if row["aggregation_type"] == "pooled_global")["rmse"] == pooled["rmse"]


def test_role_and_all_role_sufficient_statistics_reconcile() -> None:
    rows = []
    for role, value in (("r1", 1.), ("r2", 2.)):
        row = {"sample_id": role, "method": "m", "test_role": role, "source_family": "f", "sensor_strategy": "s", "noise_level": 0., "sensor_count_bin": "b"}
        prediction = np.full((3, 3), value); target = np.zeros((3, 3)); mask = np.ones((3, 3), bool)
        row.update(ai_evaluation.source_metrics(prediction, target, mask)); rows.append(row)
    by_role = ai_evaluation.aggregate_metrics(rows, ("method", "test_role")); overall = ai_evaluation.aggregate_metrics(rows, ("method",))
    role_pooled = [row for row in by_role if row["aggregation_type"] == "pooled_global"]
    all_pooled = next(row for row in overall if row["aggregation_type"] == "pooled_global")
    assert sum(row["source_squared_error_sum"] for row in role_pooled) == all_pooled["source_squared_error_sum"]


def test_pooled_physics_metrics_use_totals(tiny_dataset: Path) -> None:
    reader = ai_data.SyntheticDatasetReader(tiny_dataset); sample = reader[0]; reader.close()
    rows = []
    for index, scale in enumerate((0.9, 1.1)):
        row = {"sample_id": str(index), "method": "m", "test_role": "r", "source_family": "f", "sensor_strategy": "s", "noise_level": 0., "sensor_count_bin": "b"}
        row.update(ai_evaluation.source_metrics(sample["true_source"] * scale, sample["true_source"], sample["source_valid_mask"])); row.update(ai_evaluation.physics_metrics(sample["true_source"] * scale, sample)); rows.append(row)
    pooled = next(row for row in ai_evaluation.aggregate_metrics(rows, ("method",)) if row["aggregation_type"] == "pooled_global")
    assert pooled["physics_temperature_rmse"] == pytest.approx(math.sqrt(sum(row["temperature_squared_error_sum"] for row in rows) / sum(row["temperature_node_count"] for row in rows)))
    assert pooled["clean_relative_sensor_residual"] == pytest.approx(math.sqrt(sum(row["clean_sensor_squared_error_sum"] for row in rows) / sum(row["clean_sensor_reference_squared_sum"] for row in rows)))


def test_paired_bootstrap_deterministic_and_difference_convention() -> None:
    first = ai_evaluation.paired_bootstrap(np.array([1., 2., 3.]), np.array([2., 3., 4.]), repetitions=50, confidence=.9, seed=7)
    second = ai_evaluation.paired_bootstrap(np.array([1., 2., 3.]), np.array([2., 3., 4.]), repetitions=50, confidence=.9, seed=7)
    assert first == second and first["mean_paired_difference"] == -1
    assert first["confidence_interval"][0] <= first["confidence_interval"][1]


def test_mc_dropout_only_enables_dropout_and_is_seeded(tiny_dataset: Path) -> None:
    reader = ai_data.SyntheticDatasetReader(tiny_dataset); sample = reader[0]; normalization = reader.normalization; reader.close()
    config = {"architecture": {"input_channels": 4, "output_channels": 1, "base_channels": 2, "depth": 1, "dropout": .5, "upsampling": "bilinear"}, "nonnegative_policy": "none", "seed": 4}
    model = ai_model.build_model(config, normalization)
    first = ai_evaluation.mc_dropout_prediction(model, sample, normalization, torch.device("cpu"), passes=4, seed=9)
    second = ai_evaluation.mc_dropout_prediction(model, sample, normalization, torch.device("cpu"), passes=4, seed=9)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert np.any(first[1] > 0)
    assert all(not module.training for module in model.modules())


def test_batch_prediction_matches_batch_size_one(tiny_dataset: Path) -> None:
    reader = ai_data.SyntheticDatasetReader(tiny_dataset); samples = [reader[0], reader[1]]; normalization = reader.normalization; reader.close()
    config = {"architecture": {"input_channels": 4, "output_channels": 1, "base_channels": 2, "depth": 1, "dropout": 0., "upsampling": "bilinear"}, "nonnegative_policy": "none", "seed": 4}
    model = ai_model.build_model(config, normalization).eval(); device = torch.device("cpu")
    singles = [ai_evaluation.predict_sample(model, sample, normalization, device) for sample in samples]
    batched = ai_evaluation.predict_batch(model, samples, normalization, device)
    for single, batch in zip(singles, batched): assert np.allclose(single, batch, rtol=1e-6, atol=1e-6)


def test_evaluation_batch_size_controls_execution(tiny_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = evaluation_config(tiny_dataset); config["physics_evaluation_enabled"] = False; config["log_directory"] = str(tmp_path / "logs"); observed = []
    monkeypatch.setattr(ai_evaluation, "load_trained_model", lambda *args, **kwargs: object())
    def fake_batch(model, samples, normalization, device):
        observed.append(len(samples)); return [np.zeros_like(sample["true_source"]) for sample in samples]
    monkeypatch.setattr(ai_evaluation, "predict_batch", fake_batch)
    run_configs = {name: {} for name in ai_evaluation.METHODS[:3]}
    rows_two, _ = ai_evaluation._evaluate(config, run_configs); assert observed and set(observed) == {2}
    observed.clear(); config["evaluation_batch_size"] = 1
    rows_one, _ = ai_evaluation._evaluate(config, run_configs); assert set(observed) == {1}
    assert [(r["sample_id"], r["method"], r["rmse"]) for r in rows_one] == [(r["sample_id"], r["method"], r["rmse"]) for r in rows_two]


def test_tie_aware_spearman_and_constant_vectors() -> None:
    assert ai_evaluation.spearman_correlation(np.array([1, 1, 2, 3]), np.array([1, 2, 2, 4])) == pytest.approx(5 / 6)
    assert ai_evaluation.spearman_correlation(np.ones(4), np.arange(4)) is None


def test_uncertainty_per_sample_roles_coverage_and_floor(tiny_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = evaluation_config(tiny_dataset); partition = ai_evaluation.partition_validation(tiny_dataset, partition_seed=2, select_count=1, calibration_count=1)
    monkeypatch.setattr(ai_evaluation, "load_trained_model", lambda *args, **kwargs: object())
    def fake_prediction(model, sample, normalization, device, *, passes, seed):
        truth = np.asarray(sample["true_source"]); return truth + 0.01, np.zeros_like(truth)
    monkeypatch.setattr(ai_evaluation, "mc_dropout_prediction", fake_prediction)
    result = ai_evaluation._uncertainty(config, partition, {})
    assert len(result["per_sample"]) == 8 and result["test_sample_count"] == 8
    assert set(result["test_role_results"]) == set(ai_evaluation.TEST_ROLES)
    assert result["undefined_constant_vector_correlation_count"] == 8
    assert result["pooled_pixel_uncertainty_error_spearman"] is None
    assert "mean_per_sample_pixel_coverage" in result["test_role_results"]["test_id"]
    assert "sample_mean_coverage" not in result["test_role_results"]["test_id"]
    assert result["test_role_results"]["test_id"]["mean_interval_width"] > 0
    required = {"sample_id", "test_role", "mean_predictive_uncertainty", "maximum_predictive_uncertainty", "mean_absolute_error", "maximum_absolute_error", "uncertainty_error_spearman", "pixel_coverage", "mean_interval_width", "median_interval_width"}
    assert set(result["per_sample"][0]) == required


def test_deterministic_multi_role_preview_selection(tiny_dataset: Path) -> None:
    selected = ai_evaluation.select_preview_samples(tiny_dataset, ai_evaluation.TEST_ROLES, 4, 9)
    assert selected == ai_evaluation.select_preview_samples(tiny_dataset, ai_evaluation.TEST_ROLES, 4, 9)
    assert {role for role, _ in selected} == set(ai_evaluation.TEST_ROLES)
    assert len(ai_evaluation.select_preview_samples(tiny_dataset, ai_evaluation.TEST_ROLES, 1, 9)) == 1


def test_figure_truth_label_and_all_pixel_policy() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "Synthetic benchmark only" in source and "pooled_valid_pixels" in source
    assert "preview_sample_count" in source and "select_preview_samples" in source


def test_conformal_quantile_floor_and_coverage() -> None:
    assert ai_evaluation.conformal_multiplier(np.array([1., 2., 3., 4.]), .75) == 4
    with pytest.raises(ai_evaluation.EvaluationError): ai_evaluation.conformal_multiplier(np.array([]), .9)


def test_external_metadata_gate_never_opens_arrays(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"task_type": "external_heat_flux", "classical_q_target": False, "temperature_path": "does-not-exist.h5"}))
    result = ai_evaluation.external_compatibility(manifest)
    assert result["decision"] == "no-go" and result["arrays_opened"] is False and result["inference_refused"] is True


def test_evaluation_dataset_roles_and_identical_ids(tiny_dataset: Path) -> None:
    first = ai_evaluation.EvaluationDataset(tiny_dataset, "test_id")
    second = ai_evaluation.EvaluationDataset(tiny_dataset, "test_id", first.sample_ids)
    assert first.sample_ids == second.sample_ids
    first.close(); second.close()
    with pytest.raises(ai_evaluation.EvaluationError): ai_evaluation.EvaluationDataset(tiny_dataset, "train")


def test_tiny_end_to_end_recompute_creates_outputs_without_training(verification_case, monkeypatch: pytest.MonkeyPatch) -> None:
    config, partition, run_configs = verification_case
    logs = Path(config["log_directory"]); logs.mkdir(parents=True, exist_ok=True)
    (logs / "validation_partition.json").write_text(json.dumps(partition), encoding="utf-8")
    training_runs = {name: {"runtime_seconds": 1.25, "best_epoch": 1} for name in run_configs}
    (logs / "training_runs.json").write_text(json.dumps(training_runs), encoding="utf-8")
    reader = ai_data.SyntheticDatasetReader(config["dataset_directory"]); normalization = reader.normalization; reader.close()
    before = {}
    for name, run_config in run_configs.items():
        path = Path(config["checkpoint_directory"]) / name / "best.pt"
        checkpoint = torch.load(path, weights_only=False); checkpoint["model_state_dict"] = ai_model.build_model(run_config, normalization).state_dict(); torch.save(checkpoint, path)
        before[name] = path.read_bytes()
    monkeypatch.setattr(ai_evaluation, "train", lambda *_: pytest.fail("recompute-results called training"))
    result = ai_evaluation.recompute_results(config)
    output = Path(config["output_directory"])
    for filename in ("per_sample_metrics.csv", "aggregate_metrics.csv", "stratified_metrics.csv", "paired_bootstrap.json", "uncertainty_calibration.json", "uncertainty_per_sample.csv", "evaluation_summary.json", "external_compatibility.json"):
        assert (output / filename).is_file()
    assert result["training_runs"] == training_runs and result["checkpoint_verification"]["verified"]
    assert all((Path(config["checkpoint_directory"]) / name / "best.pt").read_bytes() == content for name, content in before.items())
    aggregate_header = (output / "aggregate_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "aggregation_type" in aggregate_header
    uncertainty_header = (output / "uncertainty_per_sample.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "mean_predictive_uncertainty" in uncertainty_header


def test_no_network_module_used() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "urllib" not in source and "requests" not in source and "http.client" not in source
