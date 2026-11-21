# Two-Stream Instability (WarpX)

This repository contains the original PIC simulation and plotting scripts for the two-stream instability. The 8.7 GB raw WarpX diagnostics remain at `/Users/colej/src/warpx/SIMS/TwoBeam/seed_runs` and are intentionally not stored in Git.

## Environment

Select your existing WarpX environment:

```bash
source /Users/colej/src/warpx/.venv/bin/activate
```

Alternatively, set `WARPX_PYTHON` to another WarpX Python executable before running the shell script.

## Run one simulation

```bash
python TwoBeams.py --seed 1
```

A run writes fresh diagnostics under `diags/` and archives any previous `diags/` directory with a timestamped `diags_backup_*` name.

## Run all five original seeds

```bash
./run_seeds.sh
```

The script runs seeds 1 through 5, shows live progress, writes `seed_runs/seed_<seed>/`, moves diagnostics into each run directory, and records `metadata.txt` and `warpx.log`. Run it from anywhere; it resolves the repository directory itself.

## Generate plots

To use the existing raw diagnostics from the original project, set `TWOSTREAM_DATA_ROOT`:

```bash
export TWOSTREAM_DATA_ROOT=/Users/colej/src/warpx/SIMS/TwoBeam
python CompareSeeds.py
python plot_electron.py
python plot_Efield.py
python ValidateGrowth.py
```

The scripts also default to local `seed_runs/` or `diags/` directories when `TWOSTREAM_DATA_ROOT` is not set.

- `CompareSeeds.py` writes `five_seed_field_energy_comparison.png`.
- `plot_electron.py` writes `phase_space_seed_1.png` and `phase_space_seed_2.png` in their seed-run directories.
- `plot_Efield.py` writes `field_energy.png`.
- `ValidateGrowth.py` writes `two_stream_growth_validation.png`.

Original images and the `two_stream.gif` animation are stored in `ole_imaged/`.
