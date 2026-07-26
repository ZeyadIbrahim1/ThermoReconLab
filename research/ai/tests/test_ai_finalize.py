"""Fast CPU-only tests for Phase 5 Task 5 finalization."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "ai_finalize.py"
SPEC = importlib.util.spec_from_file_location("ai_finalize", MODULE_PATH)
ai_finalize = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ai_finalize
SPEC.loader.exec_module(ai_finalize)


def config_for(root: Path) -> dict:
    return {
        "schema_version": 1, "dataset_directory": "data_external/phase5_dataset_default", "evaluation_directory": "research/ai/outputs/evaluation_default",
        "evaluation_log_directory": "research/ai/logs/task4_default", "checkpoint_directory": "research/ai/checkpoints/task4_default",
        "external_manifest": "data_external/external_pr_manifest.json", "final_output_directory": "research/ai/outputs/finalization",
        "final_log_directory": "research/ai/logs/finalization", "tracked_report_path": "research/ai/final_report.md",
        "tracked_model_card_path": "research/ai/model_card.md", "tracked_manifest_path": "research/ai/reproducibility_manifest.json",
        "tracked_status_path": "research/ai/FINAL_STATUS.md", "expected_classical_test_count": 643,
        "expected_research_test_count": 157, "expected_task4_test_count": 40, "external_decision": "no-go",
        "model_task_type": "synthetic_source", "external_task_type": "external_heat_flux",
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def metric_row(sample_id: str, role: str, method: str, error: float) -> dict:
    return {
        "sample_id": sample_id, "method": method, "test_role": role, "source_family": "one_gaussian", "sensor_strategy": "random", "noise_level": 0.0, "sensor_count_bin": "[0,10)",
        "rmse": error, "mae": error, "relative_l2": error, "maximum_absolute_error": error,
        "source_squared_error_sum": 4 * error ** 2, "source_absolute_error_sum": 4 * error,
        "source_target_squared_sum": 4.0, "source_valid_node_count": 4, "source_global_maximum_absolute": error,
        "temperature_squared_error_sum": 4 * (error / 10) ** 2, "temperature_node_count": 4,
        "physics_temperature_rmse": error / 10, "physics_temperature_mae": error / 10,
        "clean_sensor_squared_error_sum": 2 * (error / 20) ** 2, "clean_sensor_count": 2,
        "clean_sensor_residual_rms": error / 20, "noisy_sensor_squared_error_sum": 2 * (error / 15) ** 2,
        "noisy_sensor_count": 2, "noisy_measurement_residual_rms": error / 15,
    }


def aggregate_rows(per_sample: list[dict]) -> list[dict]:
    output = []
    for method in ai_finalize.METHODS:
        method_rows = [row for row in per_sample if row["method"] == method]
        for role in (*ai_finalize.TEST_ROLES, "all_test_roles"):
            group = method_rows if role == "all_test_roles" else [row for row in method_rows if row["test_role"] == role]
            derived = ai_finalize._derived_metrics([{key: str(value) for key, value in row.items()} for row in group])
            output.append({"method": method, "test_role": role, "aggregation_type": "mean_per_sample", "sample_count": len(group), **{key: sum(float(row[key]) for row in group) / len(group) for key in ("rmse", "mae", "relative_l2", "maximum_absolute_error", "physics_temperature_rmse")}})
            output.append({"method": method, "test_role": role, "aggregation_type": "pooled_global", "sample_count": len(group), **derived})
    return output


@pytest.fixture
def tiny_artifacts(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path; config = config_for(root); dataset = root / config["dataset_directory"]; dataset.mkdir(parents=True)
    roles = {"train": ["train-1", "train-2"], "validation": ["val-select", "val-cal"], **{role: [f"{role}-1"] for role in ai_finalize.TEST_ROLES}}
    samples = []; index = 0
    for role, ids in roles.items():
        for sample_id in ids: samples.append({"storage_index": index, "sample_id": sample_id, "split": role}); index += 1
    configuration = {"schema_version": 2, "name": "tiny"}; normalization = {"schema_version": 2, "method": "global_standard"}
    write_json(dataset / "configuration.json", configuration); write_json(dataset / "normalization.json", normalization)
    (dataset / "synthetic_dataset.h5").write_bytes(b"not-opened-as-hdf5")
    manifest = {"schema_version": 2, "sample_count": len(samples), "samples": samples, "dataset_sha256": ai_finalize.sha256_stream(dataset / "synthetic_dataset.h5"), "configuration_sha256": hashlib.sha256(ai_finalize.canonical_json(configuration)).hexdigest(), "normalization_sha256": hashlib.sha256(ai_finalize.canonical_json(normalization)).hexdigest()}
    manifest["manifest_content_sha256"] = ai_finalize._manifest_content_hash(manifest); write_json(dataset / "manifest.json", manifest)
    logs = root / config["evaluation_log_directory"]; partition = {"train_sample_ids": roles["train"], "validation_select_sample_ids": ["val-select"], "validation_calibration_sample_ids": ["val-cal"], "partition_sha256": "p" * 64}; write_json(logs / "validation_partition.json", partition)
    test_manifest = {role: ids for role, ids in roles.items() if role in ai_finalize.TEST_ROLES}; write_json(logs / "test_sample_ids.json", test_manifest)
    hashes = {"dataset_manifest_hash": manifest["manifest_content_sha256"], "dataset_hdf5_hash": manifest["dataset_sha256"], "configuration_hash": manifest["configuration_sha256"], "normalization_hash": manifest["normalization_sha256"]}
    verification = {"verified": True, "dataset_hashes": hashes, "models": {}}
    training = {}
    checkpoints = root / config["checkpoint_directory"]
    for model_index, name in enumerate(ai_finalize.LEARNED_METHODS):
        architecture = {"attention": name != "residual_no_attention", "prediction_mode": "direct" if name == "direct_sparse_mask" else "residual", "input_channel_mask": [1, 1, 0, 0] if name == "direct_sparse_mask" else [1, 1, 1, 1]}
        payload = {"epoch": model_index + 1, "best_epoch": model_index + 1, "model_architecture": architecture, "nonnegative_policy": "softplus", "train_sample_ids": roles["train"], "validation_sample_ids": ["val-select"], **hashes, "torch_version": torch.__version__, "cuda_build": None, "device_name": "CPU"}
        path = checkpoints / name / "best.pt"; path.parent.mkdir(parents=True); torch.save(payload, path)
        verification["models"][name] = {"verified": True, "epoch": model_index + 1, "best_epoch": model_index + 1}; training[name] = {"parameter_count": 10 + model_index}
    write_json(logs / "checkpoint_verification.json", verification); write_json(logs / "training_runs.json", training)
    evaluation = root / config["evaluation_directory"]; per_sample = []
    for sample_index, role in enumerate(ai_finalize.TEST_ROLES):
        for method_index, method in enumerate(ai_finalize.METHODS): per_sample.append(metric_row(test_manifest[role][0], role, method, 0.1 * (method_index + 1) + 0.01 * sample_index))
    aggregate = aggregate_rows(per_sample); write_csv(evaluation / "per_sample_metrics.csv", per_sample); write_csv(evaluation / "aggregate_metrics.csv", aggregate)
    overall = [{**row} for row in aggregate if row["test_role"] == "all_test_roles"]; write_json(evaluation / "evaluation_summary.json", {"overall_metrics": overall})
    primary = {row["sample_id"]: row["rmse"] for row in per_sample if row["method"] == "full_residual_attention"}; smooth = {row["sample_id"]: row["rmse"] for row in per_sample if row["method"] == "smoothness"}; differences = [primary[key] - smooth[key] for key in sorted(primary)]; mean = sum(differences) / len(differences)
    write_json(evaluation / "paired_bootstrap.json", {"all_test_roles/full_residual_attention/smoothness/rmse": {"difference_convention": "learned metric - baseline metric", "mean_paired_difference": mean, "confidence_interval": [mean, mean], "win_rate": 1.0, "sample_count": 4}})
    uncertainty_rows = [{"sample_id": test_manifest[role][0], "test_role": role, "pixel_coverage": 0.9, "mean_predictive_uncertainty": 0.1, "maximum_predictive_uncertainty": 0.2, "mean_absolute_error": 0.1, "maximum_absolute_error": 0.2, "uncertainty_error_spearman": 0.5, "mean_interval_width": 0.2, "median_interval_width": 0.2} for role in ai_finalize.TEST_ROLES]
    write_csv(evaluation / "uncertainty_per_sample.csv", uncertainty_rows)
    role_results = {role: {"sample_count": 1, "pixel_coverage": 0.9, "mean_per_sample_pixel_coverage": 0.9, "pooled_pixel_uncertainty_error_spearman": 0.5, "mean_per_sample_uncertainty_error_spearman": 0.5} for role in ai_finalize.TEST_ROLES}
    write_json(evaluation / "uncertainty_calibration.json", {"target_coverage": 0.9, "calibration_sample_count": 1, "calibration_pixel_count": 4, "test_sample_count": 4, "multiplier": 2.0, "pooled_pixel_uncertainty_error_spearman": 0.5, "mean_per_sample_uncertainty_error_spearman": 0.5, "test_role_results": role_results})
    external = {"decision": "no-go", "arrays_opened": False, "inference_refused": True, "model_task_type": "synthetic_source", "external_task_type": "external_heat_flux", "classical_q_target": False}; write_json(evaluation / "external_compatibility.json", external)
    for name in ("stratified_metrics.csv",): (evaluation / name).write_text("value\n1\n", encoding="utf-8")
    for name in ("run_configuration.json", "environment.json"): write_json(evaluation / name, {"tiny": True})
    external_manifest = root / config["external_manifest"]; write_json(external_manifest, {"metadata_only": True})
    research = root / "research/ai"; research.mkdir(parents=True, exist_ok=True)
    for name in ("ai_data.py", "ai_model.py", "ai_evaluation.py", "ai_finalize.py", "README.md", "requirements-ml.txt"): (research / name).write_text("optional synthetic research\n", encoding="utf-8")
    write_json(research / "configs/finalization.json", config); (research / "tests").mkdir(parents=True); (research / "tests/test_ai_finalize.py").write_text("# test\n")
    (root / "README.md").write_text("[research](research/ai/README.md) [report](research/ai/final_report.md) [card](research/ai/model_card.md) Datasets and checkpoints are not distributed. External validation remains unresolved.\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname = "tiny"\nversion = "0.1.0"\n', encoding="utf-8")
    package = root / "src/thermoreconlab"; package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    (package / "reconstruction.py").write_text("def reconstruct_tikhonov():\n    pass\n", encoding="utf-8")
    (package / "reporting.py").write_text("# protected Phase 4 reporting module\n", encoding="utf-8")
    (root / "tests").mkdir(); (root / "tests/test_reporting.py").write_text("# protected Phase 4 test\n", encoding="utf-8")
    examples = root / "examples"; (examples / "data").mkdir(parents=True)
    for name in ("01_synthetic_benchmark.py", "02_user_sensor_data.py", "03_parameter_studies.py", "04_final_demo.py"):
        (examples / name).write_text("# protected Phase 4 example\n", encoding="utf-8")
    (examples / "data/demo_sensor_measurements.csv").write_text("x,y,value\n0,0,1\n", encoding="utf-8")
    return root, config


def _mock_protected_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    staged: set[str] | None = None,
    unstaged: set[str] | None = None,
    untracked: set[str] | None = None,
) -> None:
    states = {
        "staged": set() if staged is None else staged,
        "unstaged": set() if unstaged is None else unstaged,
        "untracked": set() if untracked is None else untracked,
    }

    def fake_git_lines(root: Path, arguments: list[str]) -> list[str]:
        del root
        if arguments[:3] == ["diff", "--cached", "--name-only"]:
            return sorted(states["staged"])
        if arguments[:2] == ["diff", "--name-only"]:
            return sorted(states["unstaged"])
        if arguments[:3] == ["ls-files", "--others", "--exclude-standard"]:
            return sorted(states["untracked"])
        raise AssertionError(f"Unexpected Git command: {arguments}")

    monkeypatch.setattr(ai_finalize, "_git_lines", fake_git_lines)


def test_configuration_validation(tmp_path: Path) -> None:
    config = config_for(tmp_path); assert ai_finalize.validate_finalization_config(config) is config


def test_clean_protected_state_passes(tiny_artifacts, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = tiny_artifacts
    _mock_protected_state(monkeypatch)
    result = ai_finalize.protected_scope_audit(root)
    assert result["task5_changes_confined_to_allowed_paths"] is True
    assert result["meaningful_changed_paths"] == []


@pytest.mark.parametrize("category", ["unstaged", "staged", "untracked"])
def test_allowed_paths_pass_in_any_git_category(tiny_artifacts, monkeypatch: pytest.MonkeyPatch, category: str) -> None:
    root, _ = tiny_artifacts
    states = {category: set(ai_finalize.APPROVED_CLASSICAL_RELEASE_PATHS)}
    _mock_protected_state(monkeypatch, **states)
    result = ai_finalize.protected_scope_audit(root)
    assert result["task5_changes_confined_to_allowed_paths"] is True
    assert set(result["meaningful_git_category_paths"][category]) == ai_finalize.APPROVED_CLASSICAL_RELEASE_PATHS


def test_mixed_allowed_git_categories_pass(tiny_artifacts, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = tiny_artifacts
    _mock_protected_state(
        monkeypatch,
        staged={"pyproject.toml", "examples/01_synthetic_benchmark.py"},
        unstaged={"src/thermoreconlab/reporting.py", "tests/test_reporting.py"},
        untracked={"examples/02_user_sensor_data.py", "examples/03_parameter_studies.py"},
    )
    result = ai_finalize.protected_scope_audit(root)
    assert result["task5_changes_confined_to_allowed_paths"] is True


@pytest.mark.parametrize("unexpected", [
    "src/thermoreconlab/extra.py",
    "tests/test_extra.py",
    "examples/05_extra.py",
    "pyproject.toml.bak",
])
def test_unexpected_protected_path_fails(tiny_artifacts, monkeypatch: pytest.MonkeyPatch, unexpected: str) -> None:
    root, _ = tiny_artifacts
    _mock_protected_state(monkeypatch, unstaged={unexpected})
    with pytest.raises(ai_finalize.FinalizationError, match=unexpected.replace(".", r"\.")):
        ai_finalize.protected_scope_audit(root)


def test_generated_cache_changes_are_ignored(tiny_artifacts, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = tiny_artifacts
    generated = {
        "src/thermoreconlab/__pycache__/extra.cpython-310.pyc",
        "tests/cache.pyo",
        "examples/__pycache__/example.py",
    }
    _mock_protected_state(monkeypatch, staged={next(iter(generated))}, unstaged=generated, untracked=generated)
    result = ai_finalize.protected_scope_audit(root)
    assert result["meaningful_changed_paths"] == []
    assert set(result["ignored_generated_cache_paths"]) == generated


def test_missing_required_release_file_fails(tiny_artifacts, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = tiny_artifacts
    (root / "examples/04_final_demo.py").unlink()
    _mock_protected_state(monkeypatch)
    with pytest.raises(ai_finalize.FinalizationError, match="examples/04_final_demo.py"):
        ai_finalize.protected_scope_audit(root)


def test_protected_phase4_files_are_in_manifest_inventory(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root); manifest = ai_finalize.create_manifest(config, context, root)
    records = {record["path"]: record for record in manifest["artifacts"]}
    protected = set(ai_finalize.APPROVED_CLASSICAL_RELEASE_PATHS)
    assert protected <= records.keys()
    assert all(records[path]["generated_by_phase"] == 4 for path in protected)


def test_protected_file_hash_change_fails_manifest_verification(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root); manifest = ai_finalize.freeze(config, context, root); ai_finalize.build_reports(config, context, manifest, root)
    (root / "src/thermoreconlab/reporting.py").write_text("changed protected content\n", encoding="utf-8")
    with pytest.raises(ai_finalize.FinalizationError, match="verification failed"):
        ai_finalize.verify_manifest(config, root)


def test_final_status_distinguishes_research_test_counts(tiny_artifacts) -> None:
    root, config = tiny_artifacts; status = ai_finalize.render_final_status(ai_finalize.load_and_audit(config, root))
    assert "Research suite, Tasks 2–4: 157 passed" in status
    assert "Combined research suite including Task 5: 221 passed" in status
    assert "Task 5 finalization suite: 64 passed" in status


def test_unknown_configuration_key_rejected(tmp_path: Path) -> None:
    config = config_for(tmp_path); config["extra"] = 1
    with pytest.raises(ai_finalize.FinalizationError, match="unknown"): ai_finalize.validate_finalization_config(config)


@pytest.mark.parametrize("value", ["C:/machine/file", "/absolute/file", "../escape", "a/../escape"])
def test_portable_relative_paths(value: str, tmp_path: Path) -> None:
    config = config_for(tmp_path); config["dataset_directory"] = value
    with pytest.raises(ai_finalize.FinalizationError): ai_finalize.validate_finalization_config(config)


def test_streaming_sha256(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"; path.write_bytes(b"abcdef" * 1000)
    assert ai_finalize.sha256_stream(path, block_size=7) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_stable_path_normalization() -> None:
    assert ai_finalize.normalize_relative_path(r"research\ai\file.json") == "research/ai/file.json"


def test_canonical_manifest_hash_excludes_timestamp() -> None:
    first = {"generated_at_utc": "a", "verification": {"manifest_content_sha256": "old"}, "value": 1}; second = deepcopy(first); second["generated_at_utc"] = "b"
    assert ai_finalize.canonical_manifest_hash(first) == ai_finalize.canonical_manifest_hash(second)


def test_dataset_split_reconciliation(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root)
    assert context["dataset"]["dataset_total"] == 8 and context["dataset"]["test_total"] == 4


def test_per_sample_row_counts_and_identical_ids(tiny_artifacts) -> None:
    root, config = tiny_artifacts; result = ai_finalize.load_and_audit(config, root)["per_sample"]
    assert result == {"unique_test_sample_count": 4, "method_count": 5, "row_count": 20, "role_counts": {role: 1 for role in ai_finalize.TEST_ROLES}}


def test_duplicate_metric_row_rejected(tiny_artifacts) -> None:
    root, config = tiny_artifacts; _, rows = ai_finalize._read_csv(root / config["evaluation_directory"] / "per_sample_metrics.csv"); rows.append(dict(rows[0]))
    test_ids = json.loads((root / config["evaluation_log_directory"] / "test_sample_ids.json").read_text())
    with pytest.raises(ai_finalize.FinalizationError, match="Duplicate"): ai_finalize.reconcile_per_sample(rows, test_ids)


def test_aggregate_count_and_pooled_metrics(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root); result = context["aggregate"]
    assert result["row_count"] == 50
    primary = result["overall"]["full_residual_attention"]["pooled_global"]
    assert primary["rmse"] == pytest.approx(math.sqrt(sum((0.1 + 0.01 * i) ** 2 for i in range(4)) / 4))
    assert primary["relative_l2"] == primary["rmse"] and primary["maximum_absolute_error"] == 0.13
    assert primary["physics_temperature_rmse"] == pytest.approx(primary["rmse"] / 10)


def test_bootstrap_reconciliation(tiny_artifacts) -> None:
    root, config = tiny_artifacts; result = ai_finalize.load_and_audit(config, root)["bootstrap"]
    assert result["mean_paired_difference"] == pytest.approx(-0.4) and result["win_rate"] == 1


def test_uncertainty_counts_disjointness_and_coverage(tiny_artifacts) -> None:
    root, config = tiny_artifacts; result = ai_finalize.load_and_audit(config, root)["uncertainty"]
    assert result["calibration_sample_count"] == 1 and result["test_sample_count"] == 4
    assert all(value["pixel_coverage"] == 0.9 for value in result["coverage_by_role"].values())


def test_external_no_go(tiny_artifacts) -> None:
    root, config = tiny_artifacts; result = ai_finalize.load_and_audit(config, root)["external_compatibility"]
    assert result["decision"] == "no-go" and result["arrays_opened"] is False and result["inference_refused"] is True


def test_missing_artifact_rejected(tiny_artifacts) -> None:
    root, config = tiny_artifacts; (root / config["evaluation_directory"] / "environment.json").unlink()
    with pytest.raises(ai_finalize.FinalizationError, match="Required artifact"): ai_finalize.load_and_audit(config, root)


def test_dataset_hash_mismatch_rejected(tiny_artifacts) -> None:
    root, config = tiny_artifacts; (root / config["dataset_directory"] / "synthetic_dataset.h5").write_bytes(b"corrupt")
    with pytest.raises(ai_finalize.FinalizationError, match="HDF5 hash mismatch"): ai_finalize.load_and_audit(config, root)


def test_manifest_generation_has_required_sections_and_no_absolute_paths(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root); manifest = ai_finalize.create_manifest(config, context, root)
    required = {"schema_version", "project", "package_version", "generated_at_utc", "git_commit", "git_dirty", "python", "classical_package", "phase5_scope", "dataset", "partitions", "models", "evaluation", "uncertainty", "external_compatibility", "artifacts", "commands", "limitations", "verification"}
    assert required == set(manifest)
    assert all(not Path(record["path"]).is_absolute() and "\\" not in record["path"] for record in manifest["artifacts"])


def test_manifest_freeze_and_verification(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root); manifest = ai_finalize.freeze(config, context, root); ai_finalize.build_reports(config, context, manifest, root)
    result = ai_finalize.verify_manifest(config, root); assert result["verified"] and result["artifact_count"] == len(manifest["artifacts"])


def test_manifest_artifact_hash_mismatch(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root); manifest = ai_finalize.freeze(config, context, root); ai_finalize.build_reports(config, context, manifest, root)
    (root / config["tracked_report_path"]).write_text("changed")
    with pytest.raises(ai_finalize.FinalizationError, match="verification failed"): ai_finalize.verify_manifest(config, root)


@pytest.mark.parametrize("claim", ["smoothness Tikhonov outperformed all learned models", "attention was uniformly beneficial", "external validation succeeded", "MC dropout is a Bayesian posterior", "the model is production-ready", "Task 3 does not exist", "Task 4 does not exist", "The model predicts external heat flux", "Numerical results are real-world measurements", "Checkpoints are distributed with the package"])
def test_stale_documentation_claim_detection(claim: str) -> None:
    assert ai_finalize.detect_stale_claims({"doc.md": claim})


def test_negated_limitations_are_not_stale() -> None:
    text = "MC dropout is not a Bayesian posterior. The model is not production-ready."
    assert ai_finalize.detect_stale_claims({"doc.md": text}) == []


def test_report_model_card_and_status_rendering(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root)
    report, card, status = ai_finalize.render_final_report(context), ai_finalize.render_model_card(context), ai_finalize.render_final_status(context)
    assert all(title in report for title in ("## 1. Executive summary", "## 21. Final conclusion", "Mean per-sample", "Pooled-global"))
    assert "Synthetic research model only" in card and "Not validated on external vehicle-fire data" in card
    assert "Tasks completed: 5/5" in status and "Package version: 0.1.0" in status


def test_checkpoint_research_artifact_boundaries(tiny_artifacts) -> None:
    root, config = tiny_artifacts; card = ai_finalize.render_model_card(ai_finalize.load_and_audit(config, root))
    assert "not required by normal package users" in card and "not distributed with the package" in card
    assert "No external-data-trained checkpoint exists" in card and "human licensing review" in card


def test_final_metrics_and_figure_truth_label(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root); manifest = ai_finalize.freeze(config, context, root); ai_finalize.build_reports(config, context, manifest, root)
    output = root / config["final_output_directory"]
    _, rows = ai_finalize._read_csv(output / "final_metrics.csv"); assert len(rows) == 10
    assert (output / "final_results_overview.png").stat().st_size > 0 and ai_finalize.TRUTH_LABEL in MODULE_PATH.read_text(encoding="utf-8")


def test_documentation_numerical_consistency(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root); manifest = ai_finalize.freeze(config, context, root); ai_finalize.build_reports(config, context, manifest, root)
    result = ai_finalize.documentation_audit(context, config, root); assert result["passed"] and result["numerical_values_reconciled"]


def test_isolation_audit_on_real_repository() -> None:
    result = ai_finalize.isolation_audit(Path(__file__).parents[3]); assert result["passed"]
    assert result["checks"]["main_dependencies_include_torch"] is False and result["checks"]["package_init_imports_research"] is False


def test_no_forbidden_execution_paths_or_network() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("from ai_model", "import ai_model", "from ai_data", "import ai_data", "build_synthetic_dataset", "solve_forward", "mc_dropout_prediction", "requests", "urllib", "http.client", "h5py"):
        assert forbidden not in source


def test_no_external_hdf5_open(tiny_artifacts, monkeypatch: pytest.MonkeyPatch) -> None:
    root, config = tiny_artifacts
    original = Path.open
    def guarded(self, *args, **kwargs):
        if self == root / config["external_manifest"] or self.suffix.lower() in {".h5", ".hdf5"} and self != root / config["dataset_directory"] / "synthetic_dataset.h5": pytest.fail("external HDF5 opened")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded); assert ai_finalize.load_and_audit(config, root)["external_compatibility"]["decision"] == "no-go"


def test_tiny_end_to_end_finalization(tiny_artifacts, monkeypatch: pytest.MonkeyPatch) -> None:
    root, config = tiny_artifacts
    monkeypatch.setattr(ai_finalize, "isolation_audit", lambda *_: {"passed": True, "checks": {}})
    result = ai_finalize.run_all(config, root)
    assert result["phase5_status"] == "complete" and result["manifest_verification"]["verified"]
    output = root / config["final_output_directory"]
    for name in ("final_summary.json", "final_metrics.csv", "artifact_inventory.csv", "reproducibility_commands.txt", "final_results_overview.png"): assert (output / name).is_file()
    assert result["no_retraining"] and result["no_dataset_rebuild"] and result["external_hdf5_arrays_opened"] is False


def test_exact_phase2_mapping() -> None:
    for path in ("research/ai/ai_data.py", "research/ai/tests/test_ai_data.py", "research/ai/configs/dataset_default.json", "data_external/phase5_dataset_default/synthetic_dataset.h5", "data_external/external_pr_manifest.json"):
        assert ai_finalize.artifact_phase(path) == 2


def test_exact_phase3_mapping() -> None:
    for path in ("research/ai/ai_model.py", "research/ai/tests/test_ai_model.py", "research/ai/configs/model_smoke.json", "research/ai/requirements-ml.txt"):
        assert ai_finalize.artifact_phase(path) == 3


def test_exact_phase4_mapping() -> None:
    for path in ("research/ai/ai_evaluation.py", "research/ai/tests/test_ai_evaluation.py", "research/ai/configs/evaluation_default.json", "research/ai/logs/task4_default/training_runs.json", "research/ai/outputs/evaluation_default/aggregate_metrics.csv", "research/ai/outputs/evaluation_default/overall_source_rmse.png"):
        assert ai_finalize.artifact_phase(path) == 4


def test_task4_checkpoint_phase_mapping() -> None:
    for path in ai_finalize.CHECKPOINT_NAMES.values():
        assert ai_finalize.artifact_phase(f"research/ai/checkpoints/task4_default/{path}") == 4


def test_exact_phase5_mapping() -> None:
    for path in ("research/ai/ai_finalize.py", "research/ai/tests/test_ai_finalize.py", "research/ai/configs/finalization.json", "research/ai/final_report.md", "research/ai/model_card.md", "research/ai/reproducibility_manifest.json", "research/ai/FINAL_STATUS.md", "research/ai/README.md", "README.md"):
        assert ai_finalize.artifact_phase(path) == 5


def test_unknown_artifact_path_rejected() -> None:
    with pytest.raises(ai_finalize.FinalizationError, match="Unknown Phase 5 artifact path"):
        ai_finalize.artifact_phase("research/ai/unknown.txt")


def _write_version_tree(root: Path, pyproject_version: str, init_version: str) -> None:
    (root / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{pyproject_version}"\n', encoding="utf-8")
    package = root / "src/thermoreconlab"; package.mkdir(parents=True)
    (package / "__init__.py").write_text(f'__version__ = "{init_version}"\n', encoding="utf-8")


def test_pyproject_version_parsing(tmp_path: Path) -> None:
    _write_version_tree(tmp_path, "0.1.0", "0.1.0")
    assert ai_finalize.reconcile_package_versions(tmp_path)["pyproject_version"] == "0.1.0"


def test_package_init_version_reconciliation(tmp_path: Path) -> None:
    _write_version_tree(tmp_path, "0.1.0", "0.1.0"); result = ai_finalize.reconcile_package_versions(tmp_path)
    assert result == {"pyproject_version": "0.1.0", "package_init_version": "0.1.0", "versions_consistent": True}


def test_version_mismatch_rejected(tmp_path: Path) -> None:
    _write_version_tree(tmp_path, "0.1.0", "0.2.0")
    with pytest.raises(ai_finalize.FinalizationError, match="version mismatch"):
        ai_finalize.reconcile_package_versions(tmp_path)


def test_staged_reconstruction_change_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(root, arguments):
        return subprocess.CompletedProcess(arguments, 1 if "--cached" in arguments else 0, "", "")
    monkeypatch.setattr(ai_finalize, "_git_result", fake)
    with pytest.raises(ai_finalize.FinalizationError, match="reconstruction file is modified"):
        ai_finalize.reconstruction_git_cleanliness(tmp_path)


def test_unstaged_reconstruction_change_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(root, arguments):
        return subprocess.CompletedProcess(arguments, 0 if "--cached" in arguments else 1, "", "")
    monkeypatch.setattr(ai_finalize, "_git_result", fake)
    with pytest.raises(ai_finalize.FinalizationError, match="reconstruction file is modified"):
        ai_finalize.reconstruction_git_cleanliness(tmp_path)


def test_tracked_external_hdf5_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai_finalize, "_git_lines", lambda root, arguments: ["data_external/external.h5"])
    monkeypatch.setattr(ai_finalize, "_git_result", lambda root, arguments: subprocess.CompletedProcess(arguments, 0, "", ""))
    with pytest.raises(ai_finalize.FinalizationError, match="tracking boundary failed"):
        ai_finalize.tracking_boundary_audit(tmp_path)


def test_tracked_checkpoint_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai_finalize, "_git_lines", lambda root, arguments: ["research/ai/checkpoints/model.pt"])
    monkeypatch.setattr(ai_finalize, "_git_result", lambda root, arguments: subprocess.CompletedProcess(arguments, 0, "", ""))
    with pytest.raises(ai_finalize.FinalizationError, match="tracking boundary failed"):
        ai_finalize.tracking_boundary_audit(tmp_path)


def test_approved_frozen_assets_are_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracked = sorted(ai_finalize.APPROVED_FROZEN_TRACKED_PATHS)
    monkeypatch.setattr(ai_finalize, "_git_lines", lambda root, arguments: tracked)
    monkeypatch.setattr(ai_finalize, "_git_result", lambda root, arguments: subprocess.CompletedProcess(arguments, 0, "", ""))
    result = ai_finalize.tracking_boundary_audit(tmp_path)
    assert result["only_approved_frozen_assets_tracked"]
    assert not result["unexpected_tracked_paths"]


def test_root_readme_stale_claim_audit() -> None:
    assert ai_finalize.detect_stale_claims({"README.md": "External validation succeeded."})[0]["path"] == "README.md"


@pytest.mark.parametrize("external_statement", [
    "External validation remains unresolved.",
    "The AI workflow is not externally validated.",
])
def test_root_readme_required_links(external_statement: str) -> None:
    text = f"[research](research/ai/README.md) [report](research/ai/final_report.md) [card](research/ai/model_card.md) Datasets and checkpoints are not distributed. {external_statement}"
    result = ai_finalize.validate_root_readme(text); assert not result["missing_links"] and result["external_validation_unresolved"]


def test_reproducibility_manifest_is_tracked_output(tiny_artifacts) -> None:
    root, config = tiny_artifacts; context = ai_finalize.load_and_audit(config, root); manifest = ai_finalize.freeze(config, context, root); outputs = ai_finalize.build_reports(config, context, manifest, root)
    assert config["tracked_manifest_path"] in outputs["tracked"] and config["tracked_manifest_path"] not in outputs["generated"]
