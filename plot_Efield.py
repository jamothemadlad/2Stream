
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
data_root = Path(os.environ.get("TWOSTREAM_DATA_ROOT", folder)).resolve()

data = np.loadtxt(
    data_root / "diags/reducedfiles/FieldEnergy.txt",
    comments="#"
)

steps = data[:, 0]
time_ns = data[:, 1] * 1e9
E_energy = data[:, 3]
B_energy = data[:, 4]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Full electric-field energy evolution
axes[0].plot(time_ns, E_energy, color="royalblue")
axes[0].set_xlabel("Time (ns)")
axes[0].set_ylabel("Electric-field energy (J)")
axes[0].set_title("Field Energy — Linear Scale")
axes[0].grid(alpha=0.3)

# Logarithmic scale for exponential growth
mask = E_energy > 0

axes[1].semilogy(
    time_ns[mask],
    E_energy[mask],
    color="royalblue"
)
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("Electric-field energy (J)")
axes[1].set_title("Field Energy — Log Scale")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(folder / "field_energy.png", dpi=300)

print("Final electric energy:", E_energy[-1], "J")
print("Maximum magnetic energy:", np.max(B_energy), "J")

plt.show()
