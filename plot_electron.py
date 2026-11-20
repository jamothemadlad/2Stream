"""Plot the included physical position-velocity maps; raw particle files are not needed."""
import argparse
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from twostream.forecast import load_runs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/processed')
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2])
    p.add_argument('--snapshots',type=int,default=6)
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    args=p.parse_args()
    if args.snapshots<1:p.error('--snapshots must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    for seed,run in load_runs(args.data).items():
        if seed not in args.seeds:continue
        indices=np.unique(np.linspace(0,len(run['times_s'])-1,min(args.snapshots,len(run['times_s'])),dtype=int))
        fig,axes=plt.subplots(2,len(indices),figsize=(3.2*len(indices),6),sharex=True,sharey=True,squeeze=False)
        extent=[*run['z_edges_m'][[0,-1]],*run['velocity_edges_over_c'][[0,-1]]]
        for row in [0,1]:
            vmax=float(run['phase_mass'][indices,row].max())
            for column,index in enumerate(indices):
                axes[row,column].imshow(run['phase_mass'][index,row].T,origin='lower',extent=extent,
                                       aspect='auto',cmap='viridis',vmin=0,vmax=vmax)
                axes[row,column].set_title(f"Step {run['iterations'][index]}")
                axes[row,column].set_xlabel('Position z (m)')
            axes[row,0].set_ylabel(f"{run['species'][row]}\nPhysical velocity / c")
        fig.suptitle(f'Seed {seed}: weighted position-velocity maps')
        fig.tight_layout();file=args.output/f'phase_space_seed_{seed}.png'
        fig.savefig(file,dpi=150);plt.close(fig);print('Saved:',file.resolve())


if __name__=='__main__':main()
