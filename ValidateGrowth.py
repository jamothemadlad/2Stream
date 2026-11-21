
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0
from openpmd_viewer import OpenPMDTimeSeries

folder = Path(__file__).resolve().parent
data_root = Path(os.environ.get("TWOSTREAM_DATA_ROOT", folder)).resolve()

# Original simulation parameters
n0 = 1e15
u0 = 0.1 * c
omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
L = 8 * np.pi * u0 / omega_pe

# Read existing field diagnostics
ts = OpenPMDTimeSeries(
    str(data_root / "diags/fields"),
    backend="openpmd-api",
    check_all_files=False
)

times = omega_pe * np.asarray(ts.t)
amplitudes = {3: [], 4: []}

for iteration in ts.iterations:
    Ez, info = ts.get_field(
        field="E",
        coord="z",
        iteration=int(iteration)
    )

    Ez = np.asarray(Ez).squeeze()

    # Remove duplicate periodic endpoint if nodal data
    # contains 257 points for 256 cells.
    if Ez.size == 257:
        Ez = Ez[:-1]

    if Ez.size != 256:
        raise ValueError(f"Unexpected Ez shape: {Ez.shape}")

    spectrum = np.abs(np.fft.rfft(Ez - Ez.mean())) / Ez.size

    for mode in amplitudes:
        amplitudes[mode].append(spectrum[mode])

# Initial fit window: adjust after inspecting the graph
fit_mask = (times >= 8) & (times <= 13)

fig, ax = plt.subplots(figsize=(9, 6))

for mode, values in amplitudes.items():
    values = np.asarray(values)

    ax.semilogy(times, values, label=f"Mode {mode}")

    valid = fit_mask & (values > 0)
    if valid.sum() < 3:
        raise ValueError("Not enough points in fitting window")

    slope, intercept = np.polyfit(
        times[valid], np.log(values[valid]), 1
    )

    # Cold two-stream theory with physical beam speed
    v_physical = u0 / np.sqrt(1 + (u0 / c)**2)
    a = (2 * np.pi * mode / L) * v_physical / omega_pe
    cold_growth = np.sqrt(
        max(0, np.sqrt(1 + 4*a*a) - 1 - a*a)
    )

    print(
        f"Mode {mode}: measured gamma/omega_pe = {slope:.4f}, "
        f"cold theory = {cold_growth:.4f}"
    )

    ax.semilogy(
        times[valid],
        np.exp(intercept + slope * times[valid]),
        "--",
        label=f"Mode {mode} fit: {slope:.3f}"
    )

ax.axvspan(8, 13, color="gray", alpha=0.12)
ax.set_xlabel(r"Normalised time $\omega_{pe}t$")
ax.set_ylabel(r"Electric-field Fourier amplitude (V/m)")
ax.set_title("Two-Stream Instability: Linear Growth Validation")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(folder / "two_stream_growth_validation.png", dpi=300)
plt.show()
