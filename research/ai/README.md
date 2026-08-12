# ThermoReconLab optional ML research

## Isolation and scientific scope

Phase 5 is optional research. The classical ThermoReconLab package remains the validated core, and research code is not imported by `thermoreconlab`. Task 2 provides data preparation, Task 3 provides isolated synthetic-only training, and Task 4 provides synthetic benchmark evaluation and MC-dropout predictive-dispersion analysis. Physics-guided loss and external-data training do not exist.

The two pipeline branches have different scientific targets:

- Synthetic samples use `task_type="synthetic_source"`. The stored `true_source` is the effective source used by the classical forward model: it is defined on interior nodes, and all four boundaries are explicitly zeroed for the homogeneous-Dirichlet model. One reusable `source_valid_mask` identifies those interior target nodes.
- The E-TM-F/PR adapter uses `task_type="external_heat_flux"` and `usage_role="external_audit"`. Its target is an HFITS-derived external reference heat-flux estimate, not independent full-field ground truth and not ThermoReconLab `q`.

The only available external experiment is never split by frame. It is not training or fine-tuning data, supplies no external training normalization, and cannot demonstrate external generalization. Temporal access uses ordered frame indices only; no timestamps, physical coordinates, orientation, or gauge-to-pixel registration are inferred. The classical steady-state Poisson physics relation is not imposed on the transient experimental fields.

## Data and licensing policy

Raw external data, generated outputs, disposable streaming training logs, and resume checkpoints remain outside Git. The academic repository includes only the independently generated default synthetic dataset, its required metadata, three frozen `best.pt` checkpoints, and selected structured JSON reproducibility metadata under `logs/task4_default/`. The JSON files record partitions, checkpoint verification, and consolidated run provenance; they are intentionally committed and are not disposable training streams. Local `history.jsonl` files and `last.pt` resume checkpoints remain ignored. External HDF5 files are referenced in place and were not copied into the synthetic dataset.

The historical `reproducibility_manifest.json` describes the complete local Phase 5 closure, including generated evaluation artifacts that can be reproduced from the frozen bundle but are not all distributed through Git. The executed final notebook retains the reviewed ML evidence. Its optional live ML section requires a compatible CUDA/PyTorch environment and the corresponding local evaluation summaries; without them, the notebook skips live inference while preserving its stored outputs.

The ULRI vehicle-fire dataset is CC BY-NC 4.0. Confirm noncommercial scope, preserve attribution, and obtain human review before redistribution. HFITS is GPL-3.0 software and is scientific provenance only: no HFITS implementation code was copied, adapted, imported, or executed by this pipeline.

## Configuration, splits, and storage

`configs/dataset_smoke.json` is an 18-sample validation configuration. `configs/dataset_default.json` is a larger research configuration and never runs automatically. Configuration validation checks seeds, ranges, families, sensor capacity, OOD availability, split coverage, normalization, and HDF5 compression without solving any inverse problem.

Synthetic training uses only `regular_grid` and `random` sensor layouts. Boundary-only synthetic sensors are prohibited because homogeneous Dirichlet boundary temperatures contain no source information. `center_focused` is reserved as the OOD synthetic sensor strategy. The configured OOD shape and noise choices are likewise absent from training.

A build writes one compressed, chunked `synthetic_dataset.h5`, `configuration.json`, and `normalization.json`. It hashes and atomically publishes those artifacts first, then publishes `manifest.json` last as the completion marker. Existing final or partial artifacts are never overwritten. Validation checks all artifact hashes, schemas, HDF5 completion metadata, sample counts, source masks, sensor/index/mask consistency, and sparse measurements.

Global standardization is fitted only on `train` sample IDs. Readers open HDF5 in mode `r`, return copies, close predictably, and reject incomplete or inconsistent data. They are ordinary Python readers, not PyTorch datasets.

## Commands

Run from the repository root with the existing virtual environment.

```powershell
& ".\.venv\Scripts\python.exe" research/ai/ai_data.py inspect "C:\path\to\file.h5"

& ".\.venv\Scripts\python.exe" research/ai/ai_data.py validate-config research/ai/configs/dataset_smoke.json
& ".\.venv\Scripts\python.exe" research/ai/ai_data.py validate-config research/ai/configs/dataset_default.json

& ".\.venv\Scripts\python.exe" research/ai/ai_data.py build-smoke research/ai/configs/dataset_smoke.json --output "C:\path\to\phase5_smoke"
& ".\.venv\Scripts\python.exe" research/ai/ai_data.py build research/ai/configs/dataset_default.json --output "C:\path\to\phase5_default"

& ".\.venv\Scripts\python.exe" research/ai/ai_data.py external-manifest "C:\path\to\T_PR.h5" "C:\path\to\HF_PR.h5" "C:\path\to\external_pr_manifest.json" --experiment-id "E-TM-F" --plate-id "PR"

& ".\.venv\Scripts\python.exe" research/ai/ai_data.py preview-synthetic "C:\path\to\phase5_smoke" research/ai/outputs/dataset_preview --count 3
& ".\.venv\Scripts\python.exe" research/ai/ai_data.py preview-external "C:\path\to\external_pr_manifest.json" research/ai/outputs/dataset_preview --frame 1000 --window-size 3 --boundary-policy reject --sensor-strategy random --sensor-count 64 --sensor-seed 5102

& ".\.venv\Scripts\python.exe" research/ai/ai_data.py validate --dataset-directory "C:\path\to\phase5_smoke"
& ".\.venv\Scripts\python.exe" research/ai/ai_data.py validate --external-manifest "C:\path\to\external_pr_manifest.json"
```

External window boundary handling is mandatory: `reject`, `edge-repeat`, or `reflect`. External `boundary` sampling, when explicitly requested, is only a simulated image-layout experiment and is unrelated to the prohibited synthetic Dirichlet-boundary observations. External preview axes remain pixel row/column and units remain unresolved.

Task 2 performs no download. A prior selective Task 1 verification transferred approximately 1.118 GB for only the PR pair, exceeded the earlier 250 MB access-resolution ceiling, and did not retrieve the complete 7.135 GB archive.

## Task 3 residual-attention training

Task 3 adds an isolated PyTorch system for synthetic data only. It does not train on external data and never admits `test_id`, `test_ood_shape`, `test_ood_sensor`, or `test_ood_noise` into training. Those roles remain reserved for Task 4. The 18-sample smoke run is labeled **Functional smoke run only — Not a scientific performance result**.

The four inputs are sparse measured temperature, the binary sensor mask, identity-Tikhonov source reconstruction, and smoothness-Tikhonov source reconstruction. Persisted Task 2 statistics are reused without refitting:

- sparse temperature: `(value - sparse_temperature.mean) / sparse_temperature.scale`;
- sensor mask: unchanged binary values;
- identity and smoothness source reconstructions: `(value - true_source.mean) / true_source.scale`;
- target: `(true_source - true_source.mean) / true_source.scale`.

The U-Net head predicts a normalized residual over the normalized smoothness reconstruction. Output constraints are defined relative to normalized physical zero, `(0 - true_source_mean) / true_source_scale`, rather than numeric zero in normalized space. `relu` clamps at this threshold and `softplus` is shifted by it. The final mask assigns this threshold to invalid nodes, so denormalized boundaries are exactly physical zero; `none` leaves interior values unconstrained.

Task 3 loss is supervised image-space MSE/L1/finite-difference loss on interior nodes only; it contains no PDE or physics-guided term. Epoch reports aggregate squared error, absolute error, target norm, valid nodes, gradient error, valid edges, and global maxima over the complete epoch rather than averaging batch metrics.

PyTorch remains separate from the classical package through `requirements-ml.txt`. The installed reproducible requirement is `torch==2.11.0+cu128` from the official CUDA 12.8 PyTorch index. No torchvision or torchaudio dependency is used.

```powershell
& ".\.venv\Scripts\python.exe" research/ai/ai_model.py inspect-environment
& ".\.venv\Scripts\python.exe" research/ai/ai_model.py validate-config research/ai/configs/model_smoke.json
& ".\.venv\Scripts\python.exe" research/ai/ai_model.py model-summary research/ai/configs/model_smoke.json
& ".\.venv\Scripts\python.exe" research/ai/ai_model.py train research/ai/configs/model_smoke.json
& ".\.venv\Scripts\python.exe" research/ai/ai_model.py resume research/ai/configs/model_smoke.json "C:\path\to\last.pt"
& ".\.venv\Scripts\python.exe" research/ai/ai_model.py inspect-checkpoint "C:\path\to\best.pt"
```

Checkpoints are manifest-bound and store the synthetic dataset, configuration, and normalization hashes plus exact train/validation IDs. Resume restores Python, NumPy, CPU/CUDA PyTorch RNG state, AMP GradScaler state, train-loader generator state, optimizer/scheduler state, best epoch, and early-stopping state. After training, validated `best.pt` weights generate the qualitative prediction, and summaries distinguish best-epoch from last-epoch information.

Checkpoints are atomically published and ignored by Git. Do not distribute them without licensing review: although Task 3 trains only on ThermoReconLab synthetic data, future checkpoints influenced by CC BY-NC 4.0 external data would retain unresolved redistribution and commercial-use questions. HFITS remains GPL-3.0 scientific context only; no HFITS code is present.

## Task 4 synthetic benchmark evaluation

Task 4 evaluates only the synthetic `q` task. It partitions validation deterministically into model-selection and uncertainty-calibration subsets, trains three fixed runs, and evaluates best checkpoints on four held-out synthetic test roles. “OOD” means held-out synthetic factors, not real-world generalization.

The fixed runs are the full residual-attention model, a residual model without attention, and a direct sparse-temperature/sensor-mask model. Identity and smoothness Tikhonov are explicit baselines. Source metrics use physical interior arrays. Post-hoc consistency uses the classical `solve_forward()` API only for evaluation; no physics loss is added.

`aggregate_metrics.csv` distinguishes `mean_per_sample` from `pooled_global`. The former is the arithmetic mean of complete per-sample metrics and is retained for paired bootstrap, ablation, and subgroup reporting. The latter is calculated directly from sums and counts over the physical arrays: it does not average per-sample RMSE or relative L2, and its maximum absolute errors are true global maxima. Overall and per-role metric figures use pooled-global values; figures based on paired samples or subgroups identify their per-sample aggregation.

Before evaluation or uncertainty analysis, every `best.pt` is loaded with the manifest, HDF5, dataset-configuration, and normalization hashes expected from the synthetic manifest. Verification also requires exact duplicate-free train and validation-selection IDs, excludes calibration and all test IDs, checks `epoch == best_epoch`, architecture, nonnegative policy, training dataset path, and the shared partition across all three models. The machine-readable result is `logs/task4_default/checkpoint_verification.json`; it contains provenance and results but no tensor state.

MC dropout is predictive dispersion, not a Bayesian posterior. Its multiplier is fitted only on `validation_calibration`, after the primary best checkpoint is fixed, using a finite-sample conformal-style quantile. Test splits are not used for training, early stopping, calibration, or architecture selection.

Spearman correlations use standard average ranks for ties; constant-vector correlations are reported as undefined and counted. `uncertainty_per_sample.csv` reports uncertainty, absolute error, correlation, coverage, and interval widths for every test sample, along with pooled pixel-wise, mean per-sample, and per-role correlations in the JSON report. Interval radii use `max(predictive_std, uncertainty_std_floor)`, and lower interval bounds are clipped to zero while upper bounds remain unclipped. `mean_per_sample_pixel_coverage` is the arithmetic mean of each sample's interior-pixel coverage. It equals pooled pixel coverage when every sample has the same number of valid interior pixels.

`evaluation_batch_size` controls deterministic learned-model inference while the HDF5-backed dataset stays lazy. `preview_sample_count` selects deterministic round-robin qualitative examples across the configured test roles. The uncertainty-versus-error plot represents a recorded deterministic bounded subsample of pooled valid pixels across every test role.

```powershell
& ".\.venv\Scripts\python.exe" research/ai/ai_evaluation.py validate-config research/ai/configs/evaluation_smoke.json
& ".\.venv\Scripts\python.exe" research/ai/ai_evaluation.py run-all research/ai/configs/evaluation_smoke.json
& ".\.venv\Scripts\python.exe" research/ai/ai_evaluation.py validate-config research/ai/configs/evaluation_default.json
& ".\.venv\Scripts\python.exe" research/ai/ai_evaluation.py run-all research/ai/configs/evaluation_default.json
& ".\.venv\Scripts\python.exe" research/ai/ai_evaluation.py recompute-results research/ai/configs/evaluation_default.json
```

The final correctness correction recomputes results from the unchanged existing Task 4 best checkpoints. It does not retrain models or rebuild the default dataset. The external gate remains metadata-only; no external HDF5 array is opened.

The external compatibility gate reads manifest metadata only and returns a no-go decision. It refuses inference and source metrics because E-TM-F/PR is `external_heat_flux`, is not classical `q`, has unresolved units and coordinates, represents transient experimental physics, and offers no experiment-level external generalization split. Every Task 4 figure states “Synthetic benchmark only — No external generalization claim.” In this synthetic benchmark, all three learned models outperformed identity and smoothness Tikhonov on source RMSE, but this does not establish external or real-world generalization. The attention ablation is aggregation-dependent: full residual attention had slightly lower pooled-global RMSE (`0.298259` versus `0.300672` for no attention), while no attention had slightly lower mean per-sample RMSE (`0.244219` versus `0.247481` for full attention). The results therefore do not support a claim that attention is uniformly beneficial. All results remain synthetic-only, and the external E-TM-F/PR compatibility decision remains no-go.

## Final Phase 5 status

Phase 5 is complete as an optional, isolated research workflow. See the [final scientific report](final_report.md), [model card](model_card.md), [reproducibility manifest](reproducibility_manifest.json), and concise [final status](FINAL_STATUS.md). These artifacts preserve the synthetic-only conclusions and external no-go boundary; they do not make the research workflow part of the classical package API.
