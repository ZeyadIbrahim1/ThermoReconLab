# ThermoReconLab Phase 5 final scientific report

> **Synthetic benchmark only — No external generalization claim.**

## 1. Executive summary

Phase 5 evaluated an optional, isolated ML workflow on synthetic steady-state source reconstruction. All three learned methods improved synthetic source RMSE over identity and smoothness Tikhonov, but no external or real-world generalization was established. Classical ThermoReconLab remains the validated package core.

## 2. Research question

The question was whether fixed convolutional reconstructions can improve source-field recovery from synthetic sparse temperature observations relative to two deterministic Tikhonov baselines, while preserving a strict external-data boundary.

## 3. Classical ThermoReconLab foundation

The research workflow uses outputs derived from the classical two-dimensional finite-difference formulation. It does not alter classical APIs, dependencies, or `reconstruct_tikhonov()`.

## 4. Scientific target definition

The model target is synthetic steady-state source `q`, not experimental heat flux. Source integrals reported by Task 4 are grid-sum quantities and are **not validated physical units**.

## 5. Dataset construction

The frozen dataset contains 1,200 generated samples. It is deterministic and synthetic; no external HDF5 array contributed to construction or evaluation.

## 6. Train/validation/calibration/test design

Training used 720 samples. Validation was separated into 80 model-selection and 40 uncertainty-calibration samples. The four test roles contain 360 unique samples. Synthetic OOD is **not real-world OOD**.

## 7. Model architecture

The primary residual attention U-Net uses 1,913,038 parameters, residual prediction, all four input channels, and softplus nonnegativity.

## 8. Fixed ablations

The two fixed ablations remove attention or predict directly from sparse temperature and sensor-mask channels. No-attention has 1,884,417 parameters; direct sparse-mask has 1,913,038.

## 9. Training protocol

The immutable best epochs were 20 (full attention), 52 (no attention), and 28 (direct sparse-mask). Task 5 performed no retraining and changed no checkpoint.

## 10. Source reconstruction metrics

Mean per-sample metrics average complete sample metrics. Pooled-global metrics are derived from physical-array sums, counts, denominators, and true maxima.

| Method | Pooled-global RMSE | Mean per-sample RMSE | Pooled MAE | Pooled relative L2 | Pooled physics-temperature RMSE |
|---|---:|---:|---:|---:|---:|
| full_residual_attention | 0.298259 | 0.247481 | 0.110806 | 0.406124 | 0.002312 |
| residual_no_attention | 0.300672 | 0.244219 | 0.113946 | 0.409410 | 0.001934 |
| direct_sparse_mask | 0.301754 | 0.249034 | 0.114332 | 0.410883 | 0.002148 |
| identity | 0.644536 | 0.532809 | 0.254949 | 0.877631 | 0.018124 |
| smoothness | 0.619646 | 0.517891 | 0.395973 | 0.843740 | 0.006235 |

All learned methods outperform both classical baselines on source RMSE in this synthetic benchmark. The no-attention model has the lowest mean per-sample RMSE (0.244219), while full attention has the lowest pooled-global learned-model RMSE (0.298259). Attention is therefore aggregation-dependent and not uniformly beneficial.

## 11. Physics-consistency evaluation

Physics-temperature and sensor residuals were recomputed in Task 4 and audited here from sufficient statistics. They are post-hoc synthetic consistency metrics, not a physics training loss or external validation.

## 12. Paired bootstrap analysis

For primary minus smoothness mean per-sample RMSE, using `learned metric - baseline metric`, the mean difference is -0.27041025; the deterministic 95% interval is [-0.29286236, -0.24922654], win rate 0.975, with 360 paired samples. This interval is not claimed as evidence beyond the specified deterministic paired-bootstrap procedure.

## 13. MC-dropout predictive dispersion

MC dropout is predictive dispersion, **not a Bayesian posterior**. The pooled pixel uncertainty/error Spearman correlation is 0.94197560; the mean per-sample correlation is 0.89110090.

## 14. Calibration and coverage

Calibration used 40 samples and 36,000 pixels. The target coverage was 0.90, with multiplier 4.3381553471. Coverage is not uniformly at the nominal 90%.

| Test role | Samples | Pixel coverage | Pooled pixel Spearman |
|---|---:|---:|---:|
| test_id | 120 | 0.896093 | 0.922583 |
| test_ood_shape | 84 | 0.952302 | 0.973412 |
| test_ood_sensor | 72 | 0.882438 | 0.923274 |
| test_ood_noise | 84 | 0.867765 | 0.927340 |

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

The frozen inventory covers dataset metadata and HDF5 hash, partition and test manifests, checkpoint verification and best checkpoints, every Task 4 JSON/CSV/figure, Phase 5 source/configuration files, and final tracked reports. The academic repository includes the minimal synthetic dataset and `best.pt` evaluation bundle; generated outputs, training logs, resume checkpoints, and raw external data remain outside Git.

## 21. Final conclusion

The optional learned workflow improves synthetic source RMSE over both classical baselines in this benchmark. This is not evidence of external generalization. Classical ThermoReconLab remains the validated package core, and external E-TM-F/PR compatibility remains no-go.
