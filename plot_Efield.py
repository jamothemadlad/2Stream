"""Plot a selected seed's field energy from the included reduced diagnostics."""
import argparse
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/reduced')
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/field_energy.png')
    args=p.parse_args()
    data=np.loadtxt(args.data/f'seed_{args.seed}'/'FieldEnergy.txt',comments='#',ndmin=2)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].plot(data[:,1]*1e9,data[:,3])
    valid=data[:,3]>0;axes[1].semilogy(data[valid,1]*1e9,data[valid,3])
    for ax,title in zip(axes,['Linear scale','Logarithmic scale']):
        ax.set(xlabel='Time (ns)',ylabel=r'Electric energy per area (J m$^{-2}$)',title=title)
        ax.grid(alpha=.25)
    fig.suptitle(f'Seed {args.seed}');fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=150);print('Saved:',args.output.resolve())


if __name__=='__main__':main()
