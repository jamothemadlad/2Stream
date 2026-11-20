"""Run the included-data forecasting example on a CPU, without WarpX."""
import argparse
import csv
import json
from pathlib import Path
import os

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib_cache'))
os.environ.setdefault('MPLBACKEND','Agg')
from twostream.forecast import load_runs,run_study,plot_example


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/processed')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/example')
    parser.add_argument('--components',type=int,default=32)
    parser.add_argument('--history',type=int,default=3)
    args=parser.parse_args()
    runs=load_runs(args.data)
    result=run_study(runs,components=args.components,history=args.history)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(result['report'],indent=2)+'\n')
    rows=result['report']['test_scores']
    with (args.output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for seed,prediction in result['predictions'].items():
        fig=plot_example(runs[seed],prediction)
        fig.savefig(args.output/f'seed_{seed}_forecast.png',dpi=150)
    print('Seed | Method | Relative L1 | Relative L2')
    for row in rows:
        print(f"{row['seed']:4d} | {row['method']:32s} | {row['mean_relative_L1']:.4f} | {row['mean_relative_L2']:.4f}")
    print('Results:',args.output.resolve())


if __name__=='__main__':main()
