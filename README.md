# ThermoReconLab

ThermoReconLab is a Python package for reconstructing hidden two-dimensional
heat-source fields from sparse temperature measurements or complete temperature
fields using regularized inverse methods. The deterministic classical inverse
package is the main contribution; an optional synthetic AI research extension is
kept separate. No real experimental validation or external AI generalization is
claimed.

## Scientific model and motivation

The current forward model solves the steady-state heat equation

\[
-\Delta T = q,
\]

where \(T\) is the temperature field and \(q\) is the hidden effective
heat-source field. The model uses homogeneous Dirichlet boundary conditions on
a structured two-dimensional finite-difference grid.

Heat conduction spreads source information through the temperature field.
Recovering the source from a limited number of measurements reverses that
smoothing process and is generally underdetermined or ill-posed, so the inverse
problem requires regularization.

```text
hidden source q
    ↓
heat conduction
    ↓
temperature field T
    ↓
sparse or complete measurements
    ↓
regularized inverse reconstruction
    ↓
estimated source q
```

This controlled academic setting is relevant to questions in electronics
cooling, battery thermal monitoring, and thermal anomaly or source localization,
but the package has not been experimentally validated for those applications.

## Key capabilities

| Area | Supported capabilities |
|---|---|
| Inputs | Controlled synthetic benchmarks, sparse sensor CSV/measurements, complete 2-D temperature fields |
| Classical methods | Identity Tikhonov, smoothness Tikhonov, smooth nonnegative, compact nonnegative |
| Diagnostics and studies | Source metrics when truth exists, sensor-space residuals, rank/nullity diagnostics, alpha sensitivity, noise robustness, sensor count, sensor layout and position |
| Outputs | CSV, JSON, Markdown, PNG figures, NumPy arrays |
| Optional AI | Residual attention U-Net research using sparse measurements and classical reconstructions |

Additional data utilities support CSV, TXT, NPY, and NPZ arrays. Sensor layouts
include regular, random, center-focused, and custom configurations.

## Reviewer quick start

1. Open the fully executed
   [final notebook](notebooks/demo_presentation_final.ipynb). It covers the
   problem formulation, a recoverable synthetic case, a difficult sparse case,
   method and alpha comparisons, sensor count/layout/position studies, sparse
   CSV input, complete-field input, reporting, and optional synthetic AI
   evidence.
2. Run the complete classical demonstration in
   [examples/04_final_demo.py](examples/04_final_demo.py).
3. For the optional AI scope and evidence, read the
   [research README](research/ai/README.md),
   [final report](research/ai/final_report.md),
   [model card](research/ai/model_card.md), and
   [final status](research/ai/FINAL_STATUS.md).

## Installation

ThermoReconLab requires Python 3.10 or newer. `pyproject.toml` is the canonical
package, build, and dependency definition.

Windows PowerShell:

```powershell
git clone https://github.com/ZeyadIbrahim1/ThermoReconLab.git
cd ThermoReconLab
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux/macOS:

```bash
git clone https://github.com/ZeyadIbrahim1/ThermoReconLab.git
cd ThermoReconLab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a runtime-only installation, `requirements.txt` is a convenience mirror of
the four classical runtime dependencies declared in `pyproject.toml`:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The `dev` extra adds pytest, Jupyter, and documentation tools. Optional AI
dependencies remain separate:

```bash
python -m pip install -r research/ai/requirements-ml.txt
```

That file freezes the project's CUDA 12.8 / PyTorch environment; it is not a
universal CPU or macOS AI installation. Users without a compatible environment
can use the complete classical package and inspect the stored executed AI
evidence in the final notebook.

## Minimal package usage

### Synthetic benchmark

```python
from thermoreconlab import run_synthetic_benchmark
from thermoreconlab.reporting import export_results

result = run_synthetic_benchmark(
    grid_shape=(20, 20),
    num_sensors=16,
    alpha=1e-7,
    seed=42,
)
export_results(result, "outputs/synthetic_benchmark")
```

Synthetic mode generates known truth, solves the forward problem, samples
measurements, reconstructs the source, and reports source-error metrics.

### Sparse sensor CSV

`load_sensor_csv` expects `i`, `j`, and `value` columns; optional `x` and `y`
columns may store physical coordinates.

```python
from thermoreconlab import reconstruct_from_measurements
from thermoreconlab.data import load_sensor_csv

sensors = load_sensor_csv("examples/data/demo_sensor_measurements.csv")
result = reconstruct_from_measurements(
    sensors,
    grid_shape=(20, 20),
    alpha=1e-7,
)
```

### Complete temperature field

```python
from thermoreconlab import reconstruct_from_temperature_field
from thermoreconlab.data import load_array

temperature = load_array("temperature.npy")
result = reconstruct_from_temperature_field(
    temperature,
    alpha=1e-7,
)
```

By default every interior temperature node becomes a measurement. An ordered
interior subset can instead be supplied with `sensor_indices`.

## Classical reconstruction methods

| Method | Prior or constraint | Intended role |
|---|---|---|
| Identity Tikhonov | Magnitude penalty | Fast baseline |
| Smoothness Tikhonov | Neighboring-difference prior | Spatial coherence |
| Smooth nonnegative | Smoothness with \(q \ge 0\) | Targeted successful showcase method |
| Compact nonnegative | Smoothness, nonnegativity, and compactness/sparsity | Compact isolated-source assumptions |

No method is universally best. Alpha values are configuration-dependent, and
the identity and smoothness penalty operators have different numerical scales.
Nonnegative and compact methods add modeling assumptions; they do not prove
physical correctness or unique recoverability.

## Demonstrated evidence

The following values belong to controlled synthetic configurations in the
executed notebook, not to real experimental validation.

### Recoverable targeted case

On a 20 × 20 grid with 324 interior source unknowns, 64 center-focused sensors,
2% noise, and smooth nonnegative reconstruction at \(\alpha=10^{-9}\), the
median relative source error was 18.54% and the median peak-location distance
was one grid cell. Under this controlled synthetic configuration, the method
localized the hidden source reasonably well; it did not recover it exactly.

### Difficult sparse case

With 324 source unknowns and only 16 measurements, the observation matrix had
rank 16 and nullity 308. The relative sensor residual was 0.52%, while relative
source error was 53.0%. This illustrates information insufficiency and
non-uniqueness: **a low sensor-space residual does not guarantee accurate source
recovery.**

### Method comparison

For the targeted controlled showcase:

| Method | Relative source error | Runtime |
|---|---:|---:|
| Identity Tikhonov | 66.48% | 8.8 ms |
| Smoothness Tikhonov | 36.27% | 24.3 ms |
| Smooth nonnegative | 18.54% | 83.2 ms |
| Compact nonnegative | 18.55% | 320.8 ms |

These values compare assumptions for one benchmark; they are not universal
rankings. Across the tested sensor studies, more measurements generally reduced
median error, increasing noise degraded recovery, and geometry mattered.
Center-focused sensing was strong near its favored region but fragile for
off-center sources; regular coverage gave the best tested worst-case balance
when source position was unknown.

## Reporting outputs

`thermoreconlab.reporting.export_results` can create configuration/summary JSON,
metrics CSV, figures, and a Markdown report. Scientific fields can also be saved
as NumPy arrays where applicable.

```text
output_dir/
├── metrics.csv
├── summary.json
├── report.md
└── figures/
```

Synthetic runs have known source truth and therefore include source-error
metrics and truth-based figures. For external or otherwise unknown user data,
source truth is generally unavailable: the package reports measurement-space
diagnostics and does not invent source-error metrics.

## Official examples and final notebook

- [01_synthetic_benchmark.py](examples/01_synthetic_benchmark.py) runs and
  exports a reproducible synthetic benchmark.
- [02_user_sensor_data.py](examples/02_user_sensor_data.py) demonstrates sparse
  sensor CSV ingestion and measurement-only reporting.
- [03_parameter_studies.py](examples/03_parameter_studies.py) runs alpha,
  sensor-count, repeated-noise, layout, method, and observation studies.
- [04_final_demo.py](examples/04_final_demo.py) presents recoverable, sparse
  stress, and user-measurement cases end to end.

The [final executed notebook](notebooks/demo_presentation_final.ipynb) is the
main scientific walkthrough. All 21 code cells are executed, embedded outputs
are retained, and no error outputs are present. Generated output directories
are intentionally ignored rather than tracked.

## Optional AI research extension

The AI component is an optional research extension. The classical package
remains the main ThermoReconLab contribution.

The frozen AI dataset contains 1,200 independently generated ThermoReconLab
samples. The primary residual attention U-Net receives four inputs:

1. sparse temperature;
2. sensor mask;
3. identity reconstruction;
4. smoothness reconstruction.

Its primary residual formulation is

\[
q_{\mathrm{AI}} = q_{\mathrm{smoothness}} + \text{learned correction}.
\]

Training uses supervised source MSE, source L1, and spatial-gradient losses.
There is no PDE-based physics loss; physics consistency is evaluated post-hoc
through the classical forward model. Evaluation uses Test-ID and controlled
synthetic OOD roles. The AI workflow is not externally validated, makes no
external generalization claim, and is not production-ready.

On the controlled synthetic Test-ID benchmark, the learned model achieved
approximately 60% lower mean source error than the stored identity and
unconstrained-smoothness baselines. This comparison does not establish
superiority over the recommended smooth-nonnegative classical showcase method.

The ULRI vehicle-fire E-TM-F/PR dataset was audited as an external candidate. It
contains plate-temperature and radiative heat-flux-related fields, but its
transient heat-flux-related target was not shown to be physically equivalent to
ThermoReconLab's steady-state internal source \(q\). It was not used for training
or validation, and no raw ULRI arrays are committed.

Despite its directory name,
`data_external/phase5_dataset_default/synthetic_dataset.h5` is independently
generated ThermoReconLab synthetic data. The committed frozen bundle contains
that dataset and metadata plus three `best.pt` checkpoints:

- `full_residual_attention/best.pt`;
- `residual_no_attention/best.pt`;
- `direct_sparse_mask/best.pt`.

They support frozen inference/evaluation without retraining. Resume `last.pt`
files, raw external arrays, and disposable streaming logs are excluded.
Selected JSON files under `research/ai/logs/task4_default/` are intentionally
committed structured reproducibility metadata.

See the [research README](research/ai/README.md),
[final scientific report](research/ai/final_report.md), and
[model card](research/ai/model_card.md) for exact scope and results. The
historical `reproducibility_manifest.json` records the complete local Phase 5
closure, including generated evaluation artifacts not all distributed through
Git; the professor-facing repository retains the frozen dataset, checkpoints,
selected metadata, and executed evidence.

## Testing and reproducibility

From the repository root, run the classical suite:

```bash
python -m pytest -q
```

Run the optional AI research suite separately in its compatible environment:

```bash
python -m pytest -q \
  research/ai/tests/test_ai_data.py \
  research/ai/tests/test_ai_model.py \
  research/ai/tests/test_ai_evaluation.py \
  research/ai/tests/test_ai_finalize.py
```

At the audited v0.1.0 submission revision, 643 classical tests and 229 optional
AI research tests passed. Counts describe this revision and may evolve with the
test suite.

Synthetic generation and studies use explicit seeds. The final notebook stores
its reviewed outputs, generated report trees remain ignored, and the optional AI
bundle preserves the dataset/configuration/normalization hashes, fixed best
checkpoints, partitions, and verification metadata. PyTorch is not imported by
the classical package or included in the built wheel.

## Repository structure

```text
ThermoReconLab/
├── src/thermoreconlab/        # classical package
├── tests/                     # classical tests
├── examples/                  # official user examples
├── notebooks/
│   └── demo_presentation_final.ipynb
├── research/ai/               # optional synthetic AI research
├── data_external/             # frozen synthetic AI bundle and metadata
├── requirements.txt           # classical runtime convenience mirror
├── pyproject.toml             # canonical package/build metadata
├── README.md
└── LICENSE
```

The committed HDF5 under `data_external/phase5_dataset_default/` is synthetic
ThermoReconLab data. Raw external ULRI arrays are not distributed.

## Scientific assumptions and limitations

- The model is two-dimensional and steady-state with homogeneous Dirichlet
  boundaries; source unknowns are defined on interior nodes.
- Sparse inversion can be severely underdetermined. Grid refinement creates
  more unknowns, not new measurement information.
- Sensor geometry affects identifiability, and regularization assumptions affect
  the estimate.
- Alpha is configuration-dependent; smoothness can blur peaks.
- Nonnegativity is a modeling prior, not proof of physical correctness.
- Compactness helps only when a compact or sparse-source assumption is suitable.
- A low sensor residual does not imply low source error.
- Unknown external data has no source-truth metrics.
- Synthetic validation is not real experimental validation.
- The AI workflow is synthetic-only and has no external generalization claim.
- MC dropout measures predictive dispersion, not a Bayesian posterior or a
  production uncertainty guarantee.
- Neither the classical nor AI workflow is claimed to be production-ready.

## Project status and version

ThermoReconLab is academic/research software. Version `0.1.0` focuses on
deterministic finite-difference reconstruction, controlled synthetic evidence,
external measurement workflows, diagnostics, reporting, and an isolated
optional AI research extension.

## License

ThermoReconLab is licensed under the [MIT License](LICENSE).
