# ThermoReconLab

**ThermoReconLab** is an installable Python package for reproducible 2D inverse heat-source reconstruction experiments from sparse temperature sensor measurements.

The package is being developed as a master-level Simulation Sciences semester project. It combines numerical simulation, sparse sensing, inverse problem formulation, Tikhonov regularization, validation, scientific visualization, testing, and documentation in a reusable Python package.

## Motivation

In thermal systems such as electronics cooling, battery monitoring, thermal safety systems, and anomaly detection, the location and intensity of a heat source may not be directly observable.

Because sensors cannot be placed everywhere, ThermoReconLab studies how hidden heat sources can be reconstructed from a limited number of temperature measurements.

The package investigates questions such as:

- Where is the heat source located?
- How intense is the source?
- How accurate is the reconstruction?
- How do sensor count, sensor placement, noise, and regularization affect the result?

## Main workflows

ThermoReconLab supports two main modes.

### Synthetic benchmark mode

This mode generates a known synthetic source, solves the forward heat problem, samples sparse temperature sensors, adds controlled noise, reconstructs the source, and compares the reconstruction with the known ground truth.

It is intended for validation, benchmarking, and teaching.

### User data mode

This mode loads user-provided 2D thermal data or sensor measurements, validates and preprocesses them, reconstructs a heat-source field, and exports numerical and visual results.

The minimum supported formats will be CSV, TXT, NPY, and NPZ.

## Scientific model

The project uses the steady-state 2D heat equation

```text
-ΔT = q
```

where:

- `T` is the temperature field,
- `q` is the heat-source field,
- `Δ` is the two-dimensional Laplace operator.

The equation is discretized on a structured grid using finite differences.

The inverse problem is solved using Tikhonov regularization:

```text
minimize ||Hq - y||² + α||q||²
```

where:

- `H` is the forward-sensor observation operator,
- `y` contains sparse temperature measurements,
- `α` is the regularization parameter.

## Planned package structure

The package is organized into functional layers:

- `core`: domain, grid, fields, boundaries, and numerical operators
- `data`: synthetic data, user data input/output, and preprocessing
- `sensors`: sensor placement, sampling, measurements, and noise
- `solvers`: forward and inverse solvers
- `experiments`: reusable pipelines, runners, and parameter studies
- `analysis`: metrics, validation, and sensitivity analysis
- `visualization`: heatmaps, sensor layouts, and comparison plots
- `reporting`: result tables and report export
- `utils`: shared checks, paths, logging, and random-number utilities

## MVP scope

The first version focuses on:

- two-dimensional steady-state heat conduction,
- structured rectangular grids,
- homogeneous Dirichlet boundary conditions,
- finite-difference discretization,
- sparse sensor measurements,
- Gaussian measurement noise,
- identity-based Tikhonov regularization,
- synthetic validation,
- CSV, TXT, and NumPy data input,
- Matplotlib visualizations,
- pytest tests.

## Installation

Installation instructions will be added after the initial package skeleton is complete.

## Project status

The package is currently under development.

Current version: `0.1.0`
