
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, epsilon_0

# Project directory
folder = Path(__file__).resolve().parent
data_root = Path(os.environ.get("TWOSTREAM_DATA_ROOT", folder)).resolve()

# Plasma frequency (per electron stream)
n0 = 1e15
omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))

# Seeds to compare
seeds = [1, 2, 3, 4, 5]

# Plot only the first 10,000 timesteps
max_step = 10000

fig, ax = plt.subplots(figsize=(12, 7))

for seed in seeds:

    file = (
        data_root
        / "seed_runs"
        / f"seed_{seed}"
        / "diags"
        / "reducedfiles"
        / "FieldEnergy.txt"
    )

    if not file.exists():
        raise FileNotFoundError(f"Missing data for Seed {seed}: {file}")

    data = np.loadtxt(file, comments="#", ndmin=2)

    steps = data[:, 0]
    time = data[:, 1] * omega_pe
    electric_energy = data[:, 3]

    # Restrict to 10,000 steps and positive energies
    valid = (steps <= max_step) & (electric_energy > 0)

    ax.semilogy(
        time[valid],
        electric_energy[valid],
        linewidth=1.4,
        alpha=0.85,
        label=f"Seed {seed}"
    )

# Formatting
ax.set_xlabel(r"Normalised time $\omega_{pe}t$", fontsize=13)
ax.set_ylabel("Electric-field energy (J)", fontsize=13)

ax.set_title(
    "Two-Stream Instability — Electric-Field Energy Across 5 Seeds",
    fontsize=15
)

ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.25)

plt.tight_layout()

output = folder / "five_seed_field_energy_comparison.png"

plt.savefig(output, dpi=300)
plt.show()

print(f"Saved: {output}")
