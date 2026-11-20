# Forecasting Nonlinear Two Stream Plasma Dynamics with Linear Regression and Random Forests

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jamothemadlad/2Stream/blob/main/notebooks/two_stream_forecasting.ipynb)

AP 301 project investigating one-step forecasts of each electron stream's
position-velocity distribution. The core comparison uses a shared PCA
representation, linear regression and random forest regression, with persistence
and ballistic transport as baselines.

The included dataset contains five random-seed realisations at fixed physical
parameters. The code runs directly on these compact files; **WarpX and the raw
particle diagnostics are not needed for the analysis**.

This is an initial, reproducible forecasting experiment. Numerical convergence,
bin-resolution sensitivity, reliable long rollouts and performance at other
physical settings remain open. FNO and PINO are optional future extensions.

## Run the notebook

Click **Open in Colab**, connect a standard CPU runtime and select **Runtime >
Run all**. The first cell retrieves the repository if needed. No Google Drive
mount is required because the processed data are included here.

Locally, including VS Code, use Python 3.11 or newer (Python 3.12 was tested):

```bash
git clone https://github.com/jamothemadlad/2Stream.git
cd 2Stream
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py
```

On Windows, activate with `.venv\Scripts\activate` instead. To open the notebook
in VS Code, install the Python and Jupyter extensions and select this environment
as its kernel. If the environment lacks a kernel, install `ipykernel` into it.

The analysis writes metrics and a forecast figure to `outputs/example/`.
The exact tested package versions are in `requirements.txt`; the notebook also
records the versions in its executing environment.

## Data

`data/processed/seed_1.npz` through `seed_5.npz` contain:

| Array | Meaning |
| --- | --- |
| `phase_mass` | Shape `(201, 2, 64, 256)`: time, original stream, position, physical velocity |
| `times_s`, `iterations` | Verified physical times and saved iteration numbers |
| `z_edges_m`, `velocity_edges_over_c` | Common histogram edges; velocity is expressed as `v/c` |
| `electric_field_V_m`, `magnetic_field_T` | Three components on the saved 256-cell spatial grid |
| `charge_density_C_m3`, `current_density_A_m2` | Saved rho and Jz profiles |
| `total_particle_weight` | Physical weight represented by each stream |
| `true_mean_vz_over_c`, `true_variance_vz_over_c_squared` | Moments calculated from the original particles |
| `binned_mean_vz_over_c`, `binned_variance_vz_over_c_squared` | Moments after coarse-graining |
| `excluded_weight_fraction` | Fraction outside the fixed histogram bounds |

The map value is **bin mass normalised separately for each stream**; it is not
a rendered image or a phase-space density divided by the bin widths. The
original total particle weight is retained to reconstruct physical densities.
For the 1D simulation, particle weights have inverse-area units and integrated
energies are per transverse area, despite the generic `(J)` reduced-file header.

Raw openPMD momentum is SI single-particle `p = m gamma v`. Preprocessing uses
all three momentum components to calculate physical velocity. It does not
confuse the openPMD reader's dimensionless `p/(m c)` with velocity.

The declared initial analysis grid has 64 position bins and 256 velocity bins,
with fixed velocity bounds `[-0.5 c, +0.5 c]`. Every saved snapshot retained all
particle weight inside those bounds. Coarse-graining still discards detail:
moment errors are recorded in `data/manifest.json`, and resolution sensitivity
must be investigated before treating this representation as sufficient.

The original reduced energy and field-maximum time series are also included
under `data/reduced/`. No raw particle JSON files are needed to reproduce the
included-data analysis and plots.

## Forecasting experiment

The default split is seeds **1, 2, 3 for training**, **4 for validation** and
**5 for testing**. All snapshots from a run stay together.

1. Fit a 32-component PCA on the training runs only.
2. Use three previous snapshots' PCA coefficients and their corresponding
   spatial Ez profiles as inputs.
3. Predict the next PCA coefficients and reconstruct the distribution maps.
4. Choose a linear-regression regularisation strength and random-forest settings
   using seed 4 only.
5. Compare the frozen choices on seed 5 with persistence and conservative
   periodic ballistic transport.

The linear search includes ordinary least squares and ridge-regularised linear
regression. The selected alpha is recorded. Both regression families receive
the same features. Feature scaling is fitted on training examples only.
No future field, future distribution, seed ID or absolute time is an input.
The PCA projection of a true future map is reported separately as a
**reconstruction diagnostic**, never as a model forecast.

The initial choice of 32 PCA components and a three-snapshot history is not
claimed to be optimal. Negative reconstructed mass and particle-number errors
are reported before any correction. No positivity clipping is used.

The first local test retained about 66% of the training variance. Mean relative
L1 errors on seed 5 were 0.647 for persistence, 0.625 for ballistic transport,
0.872 for linear regression and 0.886 for random forest. **This initial setup
does not outperform the baselines.** The PCA reconstruction diagnostic itself
had error 0.867, showing that the representation also needs investigation.
These are initial results, not evidence of a successful plasma surrogate.
The [initial metrics](results/initial_one_step_metrics.json) and
[example prediction](results/initial_seed_5_forecast.png) are included for comparison.

One test seed supplies one held-out trajectory, not hundreds of independent
test experiments. Stronger uncertainty statements require more independent
runs or a properly nested whole-run evaluation. Repeatedly selecting settings
using seed 5 would invalidate its role as an untouched test.

## Plot the existing data

These commands use the included compact data and do not run WarpX:

```bash
python CompareSeeds.py
python plot_electron.py --seeds 1 2 --snapshots 6
python plot_Efield.py --seed 1
python ValidateGrowth.py --seed 1
```

Figures are saved to `outputs/`. `ValidateGrowth.py` is an exploratory Fourier
amplitude fit with an approximate cold, non-relativistic reference; it is not
a convergence or physical-validation acceptance test. Its time normalisation
uses the plasma frequency of **one stream**, matching the supplied plotting
convention. A total-electron-density frequency differs by a factor of sqrt(2).

## Generate new simulations only if needed

The optional simulation driver needs a WarpX installation with 1D Python
bindings and openPMD output, plus NumPy and SciPy. These are separate from the
lightweight analysis requirements. See the [WarpX installation
instructions](https://warpx.readthedocs.io/en/26.10/install/users.html).

Use an activated WarpX environment, or pass its Python executable explicitly:

```bash
python scripts/run_seeds.py --python /path/to/warpx-environment/bin/python --check
python scripts/run_seeds.py --python /path/to/warpx-environment/bin/python --seeds 1 2 3 4 5 --steps 10000
```

`/path/to/warpx-environment/bin/python` is a placeholder for an installed
environment. `WARPX_PYTHON` can supply the same setting to `bash run_seeds.sh`.
For a short installation check, use one seed, `--steps 10` and a fresh
`--output-root` directory.

New runs go under `runs/seed_N/`, and the launcher refuses to overwrite existing
runs. Each newly generated run saves its actual script, generated native inputs,
package metadata and log. Paths are derived from this repository or CLI options;
no original author's Mac path is required. The simulation's physical PICMI
settings and diagnostic calls match the reference source. The explicit
`--archive-existing` option on `TwoBeams.py` retains the original diagnostic
archiving behaviour for deliberate directory reuse.

The physical setup uses two electron populations, 256 spatial cells, 200
computational particles per cell per stream, a Yee electromagnetic solver,
Boris pushing and periodic boundaries in the current input source. Per-stream
density is `1e15 m^-3`; PICMI drift and thermal parameters are `0.1 c` and
`0.01 c`. These are the parameters supplied to PICMI; physical velocities are
calculated from momentum when processing the output. There are no explicitly
tracked ions in this input. This is not a dedicated electrostatic-solver run.

## Recreate the compact files from raw data

```bash
python prepare_data.py --raw-root /path/to/seed_runs
```

Supply `--audit /path/to/seed_runs_audit_new.json` to verify input files against
the original audit hashes. Existing processed outputs are protected from
accidental overwrite: select a fresh `--output` directory for a new conversion.
That directory receives its own manifest; paths in each manifest are relative
to the manifest's directory.
No PCA or simulation is performed by preprocessing.

The original five runs did not archive their full per-run input decks.
`provenance/reference_source_hashes.json` identifies the current reference
scripts; it does not retroactively supply missing run provenance. The original
audit records complete data and small energy drift, but does not establish
numerical convergence. Its auxiliary `dt` and boundary-label qualifications are
retained in `provenance/data_integrity_summary.json`; analysis uses physical
timestamps rather than the auxiliary `dt=1` label.

## Development checks

The notebook's six code cells, the analysis CLI, all four plotting scripts and
five scientific unit tests passed locally on Python 3.12. A separate ten-step
WarpX smoke test also completed. The original five simulations were not rerun
or changed. See [execution evidence](provenance/portability_verification.json).
The Colab bootstrap is supplied; a live Colab session has not been tested.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## References

- [WarpX two-stream tutorial](https://blast-warpx.github.io/warpx-tutorials/a-two-stream-instability.html)
- [WarpX 26.10 parameter reference](https://warpx.readthedocs.io/en/26.10/usage/parameters.html)
- Tsai et al., [Accelerating Kinetic Simulations of Electrostatic Plasmas with Reduced-Order Modeling](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_171.pdf), 2023.

WarpX supplies the reference plasma solver. The data preparation, forecasting
comparison and evaluation in this repository form the AP 301 investigation.
