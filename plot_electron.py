
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries

# Project directory
folder = Path(__file__).resolve().parent
data_root = Path(os.environ.get("TWOSTREAM_DATA_ROOT", folder)).resolve()

# Seeds to plot
seeds = [1, 2]

# 12 snapshots: 0, 1000, 2000, ..., 11000
snapshot_targets = list(range(0, 11001, 1000))

for seed in seeds:

    run_dir = data_root / "seed_runs" / f"seed_{seed}"
    diag_path = run_dir / "diags" / "electron_track"

    if not diag_path.exists():
        raise FileNotFoundError(f"Missing diagnostics: {diag_path}")

    # Load WarpX particle diagnostics
    ts = OpenPMDTimeSeries(
        str(diag_path),
        backend="openpmd-api",
        check_all_files=False
    )

    # Select nearest available saved iterations
    snapshots = [
        int(ts.iterations[np.argmin(np.abs(ts.iterations - target))])
        for target in snapshot_targets
    ]

    # Create 4 x 3 figure
    fig, axes = plt.subplots(
        4, 3,
        figsize=(16, 16),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for ax, iteration in zip(axes, snapshots):

        # Left electron stream
        z_left, uz_left = ts.get_particle(
            ["z", "uz"],
            species="left_beam",
            iteration=iteration
        )

        # Right electron stream
        z_right, uz_right = ts.get_particle(
            ["z", "uz"],
            species="right_beam",
            iteration=iteration
        )

        # Plot left stream
        ax.scatter(
            z_left,
            uz_left,
            s=0.5,
            alpha=0.3,
            color="royalblue",
            label="Left stream"
        )

        # Plot right stream
        ax.scatter(
            z_right,
            uz_right,
            s=0.5,
            alpha=0.3,
            color="darkorange",
            label="Right stream"
        )

        ax.set_title(f"Iteration {iteration}")
        ax.set_ylim(-0.5, 0.5)
        ax.grid(True, alpha=0.2)

    # Shared axis labels
    fig.supxlabel("z position (m)", fontsize=13)
    fig.supylabel(r"$p_z/(m_e c)$", fontsize=13)

    fig.suptitle(
        f"Two-Stream Instability — Seed {seed}",
        fontsize=17
    )

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.965)
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])

    # Save each figure inside its seed folder
    output = run_dir / f"phase_space_seed_{seed}.png"

    plt.savefig(output, dpi=300)
    plt.close(fig)

    print(f"Saved: {output}")

print("Both phase-space plots completed.")
