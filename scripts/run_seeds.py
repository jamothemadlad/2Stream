"""Run seeds in separate directories using the selected WarpX Python environment."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',default=os.environ.get('WARPX_PYTHON',sys.executable))
    parser.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5])
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--output-root',type=Path,default=ROOT/'runs')
    parser.add_argument('--check',action='store_true',help='Check Python imports without running a simulation')
    args=parser.parse_args()
    python=shutil.which(args.python) or args.python
    check=subprocess.run([python,'-c','from pywarpx import picmi; import numpy, scipy; print("WarpX Python imports OK")'])
    if check.returncode:
        raise SystemExit('Select a WarpX-enabled interpreter with --python or WARPX_PYTHON; see README.')
    if args.check:return
    if args.steps<1:parser.error('--steps must be positive')
    output=args.output_root.expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.setdefault('OMP_NUM_THREADS','1')
    env['PYTHONUNBUFFERED']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    for seed in args.seeds:
        run=output/f'seed_{seed}'
        log=output/f'seed_{seed}.log'
        if run.exists() or log.exists():
            raise FileExistsError(f'Refusing to overwrite an existing run or log: {run}')
        print(f'Starting seed {seed}, {args.steps} steps -> {run}',flush=True)
        with log.open('w') as stream:
            result=subprocess.run([python,str(ROOT/'TwoBeams.py'),'--seed',str(seed),
                                   '--steps',str(args.steps),'--output',str(run)],
                                  cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'Seed {seed} failed; inspect {log}')
        shutil.move(str(log),str(run/'simulation.log'))
        print(f'Completed seed {seed}',flush=True)


if __name__=='__main__':main()
