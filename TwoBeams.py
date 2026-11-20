"""Same two-stream physics as the reference deck, with configurable run paths."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--output", type=Path)
parser.add_argument("--archive-existing", action="store_true",
                    help="Explicitly reuse a directory and archive its previous diagnostics")
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")
seed = args.seed
print(f"Running WarpX with seed {seed}", flush=True)

from pywarpx import picmi
from pywarpx.callbacks import callfromafterstep
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
import os
from scipy.constants import c, e, m_e, epsilon_0


# Always run relative to the simulation script
project_dir = (args.output if args.output is not None else
               Path(__file__).resolve().parent / "runs" / f"seed_{seed}")
project_dir = project_dir.expanduser().resolve()
if project_dir.exists() and any(project_dir.iterdir()) and not args.archive_existing:
    parser.error(f"Output already contains files: {project_dir}; choose a fresh directory")
project_dir.mkdir(parents=True, exist_ok=True)
os.chdir(project_dir)

# Archive diagnostics from the previous run
diags = project_dir / "diags"

if diags.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive = project_dir / f"diags_backup_{timestamp}"

    shutil.move(str(diags), str(archive))
    print(f"Previous diagnostics archived: {archive.name}")

print("Ready for fresh WarpX diagnostics")


n0 = 1e15
v0 = 0.1 * c
v_th = 0.01 * c
omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
L = 8 * np.pi * v0 / omega_pe

grid = picmi.Cartesian1DGrid(
    number_of_cells=[256],
    lower_bound=[-L/2],
    upper_bound=[L/2],
    lower_boundary_conditions=["periodic"],
    upper_boundary_conditions=["periodic"],
    lower_boundary_conditions_particles=["periodic"],
    upper_boundary_conditions_particles=["periodic"]
)

left_initial = picmi.UniformDistribution(
    density=n0,
    directed_velocity=[0,0,v0],
    rms_velocity=[0, 0, v_th]
)


left_beam = picmi.Species(
    particle_type="electron",
    name="left_beam",
    initial_distribution=left_initial,
    method="Boris",
    warpx_do_not_deposit=False,
    warpx_do_not_gather=False,
    warpx_do_not_push=False
)

right_initial = picmi.UniformDistribution(
    density=n0,
    directed_velocity=[0, 0, -v0],
    rms_velocity=[0, 0, v_th]
)


right_beam = picmi.Species(
    particle_type="electron",
    name="right_beam",
    initial_distribution=right_initial,
    method="Boris",
    warpx_do_not_deposit=False,
    warpx_do_not_gather=False,
    warpx_do_not_push=False
)

solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="Yee",
    cfl=0.95
)

sim = picmi.Simulation(
    solver=solver,
    warpx_random_seed=seed,
    verbose = 0
)

layout = picmi.PseudoRandomLayout(
    n_macroparticles_per_cell=200, #200 numerical macroparticles per stream, so 256 x 200 = 51200 macroparticles per strea.m.
    grid=grid
)
particle_diag = picmi.ParticleDiagnostic(
    name="electron_track",
    period=50,
    species=[left_beam, right_beam],
    data_list=["position", "momentum", "weighting"],
    write_dir="diags",
    warpx_format="openpmd"
)


# Record total electromagnetic field energy every step
field_energy = picmi.ReducedDiagnostic(
    diag_type="FieldEnergy",
    name="FieldEnergy",
    period=1
)

# Record maximum field magnitude every step
field_maximum = picmi.ReducedDiagnostic(
    diag_type="FieldMaximum",
    name="FieldMaximum",
    period=1
)

# Save spatial field profiles every 50 steps
field_diag = picmi.FieldDiagnostic(
    name="fields",
    grid=grid,
    period=50,
    data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz", "rho", "Jz"],
    write_dir="diags",
    warpx_format="openpmd"
)
particle_energy = picmi.ReducedDiagnostic(
    diag_type="ParticleEnergy",
    name="ParticleEnergy",
    period=1
)


sim.add_species(left_beam, layout=layout)
sim.add_species(right_beam, layout=layout)
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_energy)
sim.add_diagnostic(field_maximum)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(particle_energy)

# Report actual WarpX timestep progress to Bash
progress_file = os.environ.get("WARPX_PROGRESS_FILE")

if progress_file:
    @callfromafterstep
    def report_progress():
        step = int(sim.extension.warpx.getistep(0))

        if step % 100 == 0:
            Path(progress_file).write_text(str(step))

sim.step(nsteps=args.steps)

if progress_file:
    Path(progress_file).write_text(str(args.steps))

# Archive the actual input source and environment for this newly generated run.
import json
import hashlib
import importlib.metadata
import sys
shutil.copy2(Path(__file__).resolve(), project_dir / "input_script.py")
metadata = {
    "seed": seed, "steps": args.steps,
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "python_version": sys.version,
    "packages": {name: importlib.metadata.version(name)
                 for name in ["pywarpx", "numpy", "scipy"]},
    "input_parameters": {"n0_per_stream_m3": n0, "PICMI_drift_over_c": v0/c,
                         "PICMI_thermal_over_c": v_th/c},
}
(project_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Completed run: {project_dir}", flush=True)
