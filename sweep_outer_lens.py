"""Five-seed heldout robustness sweep. Same constructed two material states."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from outer_lens import run_gate


def sweep():
    cases=[]
    for noise in (0.,.001,.01):
        for seed in range(20,25):
            outcome=run_gate(seed=seed,train_noise=noise)
            for h in ('A','B'):
                for drive in ('rapid','slow','wave'):
                    cases.append({
                        'train_noise':noise,'seed':seed,'history':h,'drive':drive,
                        **outcome['test'][f'{h}:{drive}']
                    })
    groups=defaultdict(list)
    for case in cases:
        groups[(case['train_noise'],case['history'])].append(case)
    summary={
        f'noise_{noise}:history_{h}':{
            key:float(np.mean([row[key] for row in rows]))
            for key in ('dc_rmse','arx1_rmse','arx2_rmse','fir3_rmse','fir5_rmse')
        }
        for (noise,h),rows in sorted(groups.items())
    }
    return {
        'protocol':{
            'seeds':list(range(20,25)), 'train_noise':[0.,.001,.01],
            'materials':['A','B'], 'heldout_drives':['rapid','slow','wave'],
            'training_samples':2500,'test_samples':800
        },
        'summary':summary,'cases':cases,
        'limits':'All five seeds use the SAME two exact material states; no new-state generalization.'
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/outer_lens_sweep.json'))
    args=p.parse_args()
    result=sweep()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for group,values in result['summary'].items():
        print(group,' '.join(f'{name}={values[name]:.6f}'
              for name in ('arx1_rmse','arx2_rmse','fir3_rmse','fir5_rmse','dc_rmse')))
    print('Saved:',args.out)


if __name__=='__main__':
    main()
