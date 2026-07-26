# ThermoReconLab

ThermoReconLab reconstructs two-dimensional heat-source fields from sparse
sensor measurements or supplied temperature fields. It combines a
steady-state finite-difference forward model with regularized inverse
reconstruction, diagnostics, visualization, and reproducible reporting.

## Reviewer quick start

1. Open `notebooks/demo_presentation_final.ipynb`. The executed notebook
   demonstrates all three input modes, successful and difficult synthetic
   cases, method comparison, sensor studies, sparse CSV input, complete-field
   input, reporting, and optional synthetic AI inference.
2. Run the complete classical demonstration:

   ```powershell
   & ".\.venv\Scripts\python.exe" examples/04_final_demo.py
   ```

3. Run the classical test suite:

   ```powershell
   & ".\.venv\Scripts\python.exe" -m pytest -q
   ```

   A previously verified closure had 643 passing tests; this is a historical
   result, not a permanently fixed count.
4. Review `research/ai/README.md`, `research/ai/final_report.md`,
   `research/ai/model_card.md`, `research/ai/FINAL_STATUS.md`, and
   `research/ai/reproducibility_manifest.json`.
5. For optional AI inference and verification, install the separate
   dependencies from `research/ai/requirements-ml.txt`. PyTorch is not a main
   package dependency.
6. Run the optional AI tests:

   ```powershell
   & ".\.venv\Scripts\python.exe" -m pytest `
     research/ai/tests/test_ai_data.py `
     research/ai/tests/test_ai_model.py `
     research/ai/tests/test_ai_evaluation.py `
     research/ai/tests/test_ai_finalize.py -q
   ```

   A previously verified closure had 228 passing tests; this too is a
   historical result rather than a fixed requirement.
7. The repository includes a minimal frozen AI evaluation bundle: an
   independently generated ThermoReconLab synthetic dataset and required
   metadata, three best checkpoints and their required verification metadata.
   It contains no resume (`last.pt`) checkpoints and no raw external data.

Project map:

```text
src/thermoreconlab/
tests/
examples/
research/ai/
notebooks/demo_presentation_final.ipynb
```

The classical package is the main contribution. The optional AI extension is
synthetic-only: it predicts a source correction from sparse temperatures, a
sensor mask, and identity and smoothness reconstructions. It has no PDE-based
physics loss; the forward solver checks consistency after prediction. It has
no external validation and is not production-ready.

## Motivation

Heat sources may be difficult to observe directly in applications such as
electronics cooling, battery temperature monitoring, thermal anomaly
detection, and heat-source localization. Temperature sensors measure the
effect of a source rather than the source itself, and practical systems often
provide only a limited number of measurements. ThermoReconLab provides a
controlled academic environment for studying how source location and
intensity estimates depend on the measurements and modelling assumptions. It
does not claim industrial or real experimental validation.

## Main capabilities

- Synthetic heat-source and temperature generation for benchmarking.
- Regular, random, center-focused, and custom sparse sensor layouts.
- Reproducible relative Gaussian measurement noise.
- Identity and first-difference smoothness Tikhonov regularization.
- CSV ingestion for external sensor measurements.
- Reconstruction from user-provided sparse measurements.
- Reconstruction from complete temperature fields or selected interior nodes.
- Regularization, sensor-count, noise, and sensor-layout studies.
- Source-error metrics for synthetic truth and measurement-space residuals for
  every reconstruction mode.
- Observation-matrix rank, nullity, conditioning, and underdetermination
  diagnostics.
- CSV, JSON, Markdown, and Matplotlib reporting exports.
- CSV, TXT, NPY, and NPZ array input and output through the data API.

## Installation

ThermoReconLab requires Python 3.10 or newer. Install the package and its
runtime dependencies from the repository root:

```bash
python -m pip install -e .
```

The existing `dev` extra includes the project's test and development tools:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

The high-level synthetic workflow generates known truth, samples noisy
measurements, reconstructs the source, and calculates validation metrics:

```python
from pathlib import Path

from thermoreconlab import run_synthetic_benchmark
from thermoreconlab.reporting import export_results

result = run_synthetic_benchmark(
    grid_shape=(20, 20),
    source_type="two_gaussians",
    sensor_strategy="regular",
    num_sensors=16,
    noise_level=0.02,
    alpha=1e-7,
    regularization="identity",
    seed=42,
)

export_results(result, Path("outputs") / "synthetic_benchmark")
```

Source-error metrics in this mode compare the reconstruction with known
synthetic truth on interior source nodes.

## External sensor-data workflow

`load_sensor_csv` expects the columns `i`, `j`, and `value`. Optional `x` and
`y` columns may record physical coordinates. The demonstration file uses this
form:

```csv
i,j,value,x,y
1,1,0.0001483199849599157,0.05263157894736842,0.05263157894736842
1,7,0.0008665520889166847,0.05263157894736842,0.3684210526315789
```

The integer indices must be valid for the requested grid shape. The default
physical domain is the unit square, and the forward model uses homogeneous
Dirichlet boundary conditions.

```python
from pathlib import Path

from thermoreconlab import reconstruct_from_measurements
from thermoreconlab.data import load_sensor_csv
from thermoreconlab.reporting import export_results

sensor_data = load_sensor_csv(
    Path("examples/data/demo_sensor_measurements.csv")
)
result = reconstruct_from_measurements(
    sensor_data,
    grid_shape=(20, 20),
    alpha=1e-7,
    regularization="identity",
)
export_results(result, Path("outputs") / "user_measurements")
```

Ground-truth source metrics are unavailable for this workflow.

Only measurement-space residuals can be reported without an independently
known source.

## Full temperature-field workflow

`reconstruct_from_temperature_field` accepts a two-dimensional temperature
array. By default, every interior temperature node becomes a measurement; an
ordered interior subset can instead be supplied through `sensor_indices`.
This differs from loading an already sparse sensor CSV.

```python
from thermoreconlab import reconstruct_from_temperature_field
from thermoreconlab.data import load_array

temperature = load_array("temperature.npy")
result = reconstruct_from_temperature_field(
    temperature,
    alpha=1e-7,
    regularization="identity",
)
```

The returned measurement result retains the selected measurements and the
reconstructed source, but not the original complete temperature array.
Standard reporting therefore does not invent a full temperature-field plot or
truth-based source errors for this mode.

## Regularization and parameter studies

The high-level workflows support:

- `regularization="identity"` for an identity penalty;
- `regularization="smoothness"` for an unconstrained first-difference
  smoothness penalty.

These penalty operators have different numerical scales, so their alpha
values are not directly comparable. Regularization can reduce recovered
amplitudes, and an alpha value is meaningful only for its source, grid,
measurement geometry, noise, and penalty configuration.

Synthetic studies can compare tested alpha values because the true source is
known. A selected value should be described as the **best tested alpha for
this configuration**, not as universally optimal or as automatic package
tuning. `examples/03_parameter_studies.py` demonstrates regularization,
sensor-count, repeated-noise, sensor-layout, method-comparison, and
observation-matrix studies using package APIs.

The standard identity and smoothness benchmark workflows are unconstrained,
so negative reconstruction artifacts can occur. Separate nonnegative smooth
and compact reconstruction functions are available, but they do not establish
unique recoverability or uncertainty bounds.

## Reporting outputs

`export_results(result, output_dir, dpi=300)` creates:

```text
output_dir/
├── metrics.csv
├── summary.json
├── report.md
└── figures/
```

For a synthetic `ExperimentResult`, the figures directory contains:

- `true_source.png`
- `temperature.png`
- `sensor_measurements.png`
- `reconstructed_source.png`
- `error_map.png`
- `reconstruction_comparison.png`

The tables and report distinguish source-error metrics from measurement-space
residuals and include observation-matrix diagnostics.

For external-measurement and temperature-field results, standard figures are
limited to:

- `sensor_measurements.png`
- `reconstructed_source.png`

These modes contain measurement-space diagnostics but no truth-based metrics
or figures. Scientific fields can separately be saved as NumPy arrays with
`save_array`; the final demonstration shows this explicit workflow. Runtime
fields may vary between otherwise deterministic exports.

## Official examples

- `examples/01_synthetic_benchmark.py` runs and exports one synthetic
  benchmark.
- `examples/02_user_sensor_data.py` demonstrates CSV sensor-data ingestion and
  user-mode reporting.
- `examples/03_parameter_studies.py` runs deterministic parameter studies and
  observation diagnostics.
- `examples/04_final_demo.py` presents recoverable synthetic, sparse stress,
  and external-measurement cases end to end.

Run an example from the repository root after installation, for example:

```bash
python examples/04_final_demo.py
```

## Scientific assumptions and limitations

- The model is the two-dimensional steady-state heat equation on a structured
  finite-difference grid.
- Homogeneous Dirichlet boundary conditions are assumed.
- Sparse inverse reconstruction is generally underdetermined and does not
  imply unique source recoverability.
- Results depend on sensor or measurement geometry, the regularization method,
  and the configuration-specific alpha value.
- Regularization may reduce reconstructed source amplitudes.
- Unconstrained reconstructions may contain negative source artifacts.
- Residuals are measured in sensor space; a low residual does not guarantee a
  low source error.
- Synthetic data provides validation and benchmarking, not real experimental
  validation.
- The classical package does not provide uncertainty quantification. The
  optional AI research workflow evaluates MC-dropout predictive dispersion,
  not a Bayesian posterior or a production uncertainty guarantee.
- Available nonnegative solvers impose a constraint but do not prove physical
  correctness or uniqueness.

## Testing

The root pytest configuration intentionally runs the classical package tests
under `tests/`. Run that suite from the repository root with:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

A previously verified closure had 643 classical tests passing; this is a
historical reference rather than a fixed requirement as the suite may grow.

The optional AI research suite requires the dependencies listed in
`research/ai/requirements-ml.txt` and is run separately:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest `
  research/ai/tests/test_ai_data.py `
  research/ai/tests/test_ai_model.py `
  research/ai/tests/test_ai_evaluation.py `
  research/ai/tests/test_ai_finalize.py -q
```

A previously verified combined AI research closure had 228 tests passing;
this too is historical rather than a permanent expected count.

## Optional Phase 5 AI research

The deterministic classical package remains the main validated product. An
optional, isolated PyTorch research extension is trained and evaluated on
synthetic ThermoReconLab data. It uses sparse temperatures, a sensor mask, and
classical reconstructions to predict either a source field or a correction to
a classical reconstruction. The complete workflow may therefore be described
as hybrid.

Physics consistency is evaluated after prediction using the classical forward
model. The current neural-network training objective does not contain a
PDE-based physics loss. The AI workflow is not externally validated and is not
production-ready.

The workflow is documented in [research/ai/README.md](research/ai/README.md),
with its synthetic-only findings in the [final scientific report](research/ai/final_report.md)
and usage boundaries in the [model card](research/ai/model_card.md). PyTorch is
not a normal package dependency, and no research module is imported by
`thermoreconlab`. Research datasets and checkpoints are not distributed in the
installed Python package or wheel; the repository contains only the minimal
frozen synthetic evaluation bundle described above. The E-TM-F/PR external
dataset was audited but not used for training or validation: its target is
transient heat-flux-related, whereas ThermoReconLab predicts a steady-state
internal source `q`, and physical target equivalence was not established. The
committed HDF5 dataset is independently generated ThermoReconLab synthetic
data. The compatibility decision remains no-go.

## Project status

ThermoReconLab is academic/research software. Version `0.1.0` focuses on
deterministic finite-difference inverse reconstruction, controlled synthetic
evaluation, external measurement workflows, diagnostics, and reporting.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
