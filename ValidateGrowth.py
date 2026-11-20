"""Exploratory Fourier growth fits with a cold non-relativistic reference.

The reference is an approximation for these warm, relativistically pushed runs;
this plot is not a convergence or physical-validation acceptance test.
"""
import argparse
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c,e,m_e,epsilon_0
from twostream.forecast import load_runs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/processed')
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--window',type=float,nargs=2,default=[8.,13.])
    p.add_argument('--modes',type=int,nargs='+',default=[3,4])
    p.add_argument('--output',type=Path,default=ROOT/'outputs/two_stream_growth_validation.png')
    args=p.parse_args()
    run=load_runs(args.data)[args.seed]
    omega=np.sqrt(1e15*e**2/(epsilon_0*m_e));tau=run['times_s']*omega
    Ez=run['electric_field_V_m'][:,2]
    spectrum=abs(np.fft.rfft(Ez-Ez.mean(axis=1,keepdims=True),axis=1))/Ez.shape[1]
    length=np.ptp(run['z_edges_m'])
    velocity=.1*c/np.sqrt(1+.1**2)
    fig,ax=plt.subplots(figsize=(9,5))
    for mode in args.modes:
        if mode<1 or mode>=spectrum.shape[1]:p.error('Fourier mode is out of range')
        amplitude=spectrum[:,mode]
        valid=(tau>=args.window[0])&(tau<=args.window[1])&(amplitude>0)
        if valid.sum()<3:raise ValueError('Not enough saved samples in the selected fitting window')
        slope,intercept=np.polyfit(tau[valid],np.log(amplitude[valid]),1)
        a=(2*np.pi*mode/length)*velocity/omega
        cold=np.sqrt(max(0,np.sqrt(1+4*a*a)-1-a*a))
        print(f'Mode {mode}: fitted amplitude growth/wp_beam={slope:.5f}; cold approximation={cold:.5f}')
        ax.semilogy(tau,amplitude,label=f'Mode {mode}')
        ax.semilogy(tau[valid],np.exp(intercept+slope*tau[valid]),'--',label=f'Fit {mode}: {slope:.3f}')
    ax.axvspan(*args.window,color='grey',alpha=.12)
    ax.set(xlabel=r'$\omega_{p,\mathrm{beam}}t$',ylabel='Electric-field Fourier amplitude (V/m)',
           title='Exploratory amplitude growth fits; cold reference is approximate')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=150)
    print('Saved:',args.output.resolve())


if __name__=='__main__':main()
