# ThermoReconLab synthetic source model card

> **Synthetic research model only.**
> **Not validated for operational heat-source localization.**
> **Not validated on external vehicle-fire data.**

## Model name

ThermoReconLab full residual-attention synthetic source reconstruction model.

## Version/status

Phase 5 final research artifact for package version 0.1.0; not production-ready.

## Intended use

Controlled research on synthetic two-dimensional steady-state source reconstruction and comparison with fixed ablations and classical baselines.

## Out-of-scope use

Operational localization, safety decisions, experimental heat-flux prediction, vehicle-fire reconstruction, external generalization, or physical-unit inference.

## Inputs

Four normalized grid channels: sparse temperature, sensor mask, identity reconstruction, and smoothness reconstruction.

## Output

A nonnegative synthetic source-grid estimate using a softplus policy. Grid sums are not validated physical source units.

## Architecture

Residual attention U-Net, 1,913,038 parameters, architecture `{"attention": true, "base_channels": 32, "depth": 3, "dropout": 0.1, "input_channel_mask": [1, 1, 1, 1], "input_channels": 4, "output_channels": 1, "prediction_mode": "residual", "upsampling": "bilinear"}`. Ablations are residual without attention and direct sparse-mask prediction.

## Training data

720 synthetic samples only. No external-data-trained checkpoint exists.

## Evaluation data

360 held-out synthetic samples across ID and three synthetic OOD roles. Synthetic OOD is not real-world OOD.

## Metrics

Primary pooled-global source RMSE: 0.298259. Primary mean per-sample source RMSE: 0.247481. All learned methods beat identity and smoothness source RMSE here; attention is not uniformly beneficial across aggregation types.

## Uncertainty

MC-dropout predictive dispersion, not a Bayesian posterior. Target interval coverage is 0.90; coverage varies by role.

## Limitations

Synthetic-only evaluation; fixed generator and scientific settings; unresolved calibration transfer; no validated source units; no operational or production readiness.

## Ethical/licensing considerations

Incorrect localization could create safety risks if misused. Checkpoint redistribution requires human licensing review.

## External-data restriction

E-TM-F/PR is metadata-only and no-go because external heat flux is scientifically incompatible with the synthetic source `q` target. No external HDF5 array was opened.

## Reproducibility

See `research/ai/reproducibility_manifest.json`; Task 5 did not retrain models or rebuild data.

## Checkpoint information

Research artifact `research/ai/checkpoints/task4_default/full_residual_attention/best.pt`, SHA-256 `b63c3e23bea84b16f64ab28e668b1587af7a183a3321d5f6f8a9e4365c19022b`, best epoch 20. It is not required by normal package users and is not distributed with the package.
