"""Reproduce the seed energy comparison using included reduced diagnostics."""
import argparse
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e,m_e,epsilon_0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/reduced')
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5])
    p.add_argument('--output',type=Path,default=ROOT/'outputs/five_seed_field_energy_comparison.png')
    args=p.parse_args()
    omega=np.sqrt(1e15*e**2/(epsilon_0*m_e))
    fig,ax=plt.subplots(figsize=(11,6))
    for seed in args.seeds:
        path=args.data/f'seed_{seed}'/'FieldEnergy.txt'
        data=np.loadtxt(path,comments='#',ndmin=2)
        valid=data[:,3]>0
        ax.semilogy(data[valid,1]*omega,data[valid,3],label=f'Seed {seed}',lw=1.3)
    ax.set(xlabel=r'Normalised time $\omega_{p,\mathrm{beam}}t$',
           ylabel=r'Electric-field energy per transverse area (J m$^{-2}$)',
           title='Two-stream instability across five random seeds')
    ax.legend();ax.grid(alpha=.25,which='both');fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160);print('Saved:',args.output.resolve())


if __name__=='__main__':main()
