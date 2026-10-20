"""Evaluate one-bit reward writes against causal controls and a teacher reference."""
from __future__ import annotations
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import time
import subprocess
import numpy as np
from adaptive_medium import make_grid
from adaptive_medium_task import TaskConfig
from adaptive_medium_gate import (make_evaluation_sources,evaluate_material,score_episodes,
    train_material,validate_material,refinement_check,verify_lock,git_output,write_json)
from reward_medium import train_reward_material


THRESHOLDS={'accuracy':.8,'gain':.2,'erasure_fraction':.5,'refinement':.02}


def default_reward_protocol():
    return {'version':'reward-material-v1','locked':False,'config':asdict(TaskConfig()),
        'updates':512,'pairs':8,'sigma':.04,'rate':.01,'temperature':5.,'teacher_updates':600,
        'per_cue':100,'shuffle_count':20,'development_seeds':[211,223,241],
        'heldout_seeds':list(range(301,311)),'thresholds':THRESHOLDS.copy()}


def validate_reward_protocol(p):
    try:
        TaskConfig(**p['config'])
        if p['development_seeds']!=[211,223,241] or p['heldout_seeds']!=list(range(301,311)) or p['thresholds']!=THRESHOLDS:
            raise ValueError('reward seed split or verdict thresholds changed')
        for k in ('updates','pairs','teacher_updates','per_cue','shuffle_count'):
            if not isinstance(p[k],int) or isinstance(p[k],bool) or p[k]<1:raise ValueError('invalid experiment count')
        if p['pairs']<2 or p['pairs']%2:raise ValueError('pair count must be even and >=2')
        if not np.isfinite([p['sigma'],p['rate'],p['temperature']]).all() or min(p['sigma'],p['rate'],p['temperature'])<=0:
            raise ValueError('invalid reward learning parameters')
    except (KeyError,TypeError) as exc:raise ValueError('incomplete reward protocol') from exc


def summarize_reward(rows):
    if not rows:raise ValueError('no reward seed results')
    names=list(rows[0]['mappings'][0]['arms']);arms={}
    required={'reward','frozen','erased','shuffled_reward','random_updates'}
    if not required.issubset(names):raise ValueError('missing reward controls')
    for name in names:
        values=[float(np.mean([m['arms'][name]['accuracy'] for m in row['mappings']])) for row in rows]
        if not np.isfinite(values).all() or min(values)<0 or max(values)>1:raise ValueError('invalid reward metrics')
        arms[name]={'mean':float(np.mean(values)),'std':float(np.std(values)),
                    'min':min(values),'max':max(values),'per_seed':values}
    r=arms['reward']['mean'];gain=r-arms['frozen']['mean']
    checks={'accuracy':r>=.8,
        'frozen_advantage':gain>=.2-1e-12,
        'shuffled_reward_advantage':r-arms['shuffled_reward']['mean']>=.2-1e-12,
        'random_update_advantage':r-arms['random_updates']['mean']>=.2-1e-12,
        'erasure':r-arms['erased']['mean']>=.5*gain-1e-12,
        'validity':all(m['invalid_training']==0 and m['numerical_passed'] and
            all(a.get('invalid',0)==0 for a in m['arms'].values()) for row in rows for m in row['mappings'])}
    return {'arms':arms,'checks':checks,'passed':all(checks.values()),
            'failed_reasons':[k for k,v in checks.items() if not v]}


def evaluate_reward_seed(seed,p):
    cfg=TaskConfig(**p['config']);grid=make_grid(cfg.size);x,cues=make_evaluation_sources(seed,cfg,p['per_cue'])
    mappings=[]
    for mapping in ((0,1),(1,0)):
        targets=np.asarray(mapping)[cues];runs={}
        for mode in ('reward','shuffled_reward','random_updates'):
            print(f'seed {seed} mapping {mapping} arm {mode}: {p["updates"]*p["pairs"]*2} forward trials',flush=True)
            runs[mode]=train_reward_material(grid,seed,mapping,cfg,updates=p['updates'],pairs=p['pairs'],
                sigma=p['sigma'],rate=p['rate'],temperature=p['temperature'],mode=mode)
        teacher=train_material(grid,seed,mapping,cfg,updates=p['teacher_updates'])
        runs['teacher']=teacher;primary=runs['reward'];arms={};start=time.perf_counter()
        for name,run in runs.items():
            validate_material(run.final,grid,cfg)
            arms[name]=score_episodes(evaluate_material(grid,run.final,x,cfg),targets)
        for name in ('frozen','erased'):
            arms[name]=score_episodes(evaluate_material(grid,primary.initial,x,cfg),targets)
        rng=np.random.default_rng(np.random.SeedSequence([seed,80]));shuffles=[]
        for _ in range(p['shuffle_count']):
            metrics=score_episodes(evaluate_material(grid,rng.permutation(primary.final),x,cfg),targets)
            metrics.pop('scores');metrics.pop('energies');shuffles.append(metrics)
        arms['shuffled_material']={'accuracy':float(np.mean([a['accuracy'] for a in shuffles])),
            'invalid':sum(a['invalid'] for a in shuffles),'count':sum(a['count'] for a in shuffles),
            'permutations':shuffles}
        refinement=refinement_check(grid,primary.final,x[[0,p['per_cue']]],cfg)
        mappings.append({'mapping':list(mapping),'arms':arms,'initial_material':primary.initial.tolist(),
            'learned_material':primary.final.tolist(),
            'materials':{name:run.final.tolist() for name,run in runs.items()},
            'curves':{name:run.curve for name,run in runs.items()},
            'costs':{name:run.costs for name,run in runs.items()},
            'evaluation_seconds':time.perf_counter()-start,
            'refinement':refinement,'numerical_passed':refinement['passed'],
            'invalid_training':sum(run.costs.get('invalid_training_trials',0) for run in runs.values())})
        print('  accuracy '+str({k:round(v['accuracy'],4) for k,v in arms.items()}),flush=True)
    return {'seed':seed,'mappings':mappings}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','development','full'),default='development')
    parser.add_argument('--protocol',type=Path,default=Path(__file__).parent/'protocols/reward_medium.json')
    parser.add_argument('--lock-commit')
    parser.add_argument('--out',type=Path,default=Path('results/reward_medium/development.json'))
    parser.add_argument('--replay',type=Path)
    args=parser.parse_args(argv)
    try:
        p=json.loads(args.protocol.read_text());validate_reward_protocol(p)
        if args.mode=='full':
            verify_lock(args.protocol,args.lock_commit)
            for name in ('reward_medium.py','reward_gate.py'):
                git_output('cat-file','-e',f'{args.lock_commit}:{name}')
        actual=p.copy()
        if args.mode=='smoke':actual.update(updates=2,pairs=2,teacher_updates=2,per_cue=3,shuffle_count=2)
        seeds=[211] if args.mode=='smoke' else p['heldout_seeds' if args.mode=='full' else 'development_seeds']
        receipt={'protocol':p,'actual':actual,'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            'code_commit':git_output('rev-parse','HEAD'),'config':p['config'],'mode':args.mode,
            'held_out':args.mode=='full','seed_results':[],'complete':False,
            'budget_policy':f'Reward and two feedback controls use equal forward trials. Teacher uses {actual["teacher_updates"]} updates; actual forward/adjoint costs are reported, not declared compute-matched.'}
        start=time.perf_counter()
        for seed in seeds:
            receipt['seed_results'].append(evaluate_reward_seed(seed,actual));write_json(args.out,receipt)
        receipt['summary']=summarize_reward(receipt['seed_results']);receipt['complete']=True
        receipt['scientific_gate_passed']=bool(args.mode=='full' and receipt['summary']['passed'])
        receipt['seconds']=time.perf_counter()-start;write_json(args.out,receipt)
        if args.replay:
            from adaptive_medium_replay import replay_episodes,write_replay
            write_replay(args.replay,grid=make_grid(TaskConfig(**p['config']).size),receipt=receipt,episodes=replay_episodes(receipt))
            text=args.replay.read_text().replace('before and after supervised learning','before and after learning from one-bit rewards')
            text=text.replace('This is a supervised two-cue association task.','This two-cue task was learned from scalar rewards.')
            text=text.replace('This replay shows actual recorded activity', 'This replay illustrates the first evaluation seed and shows actual recorded activity')
            text=text.replace('A six-parameter linear classifier is a strong, cheaper reference.','The teacher-gradient reference is a separate arm with its actual simulation cost reported.')
            args.replay.write_text(text)
        print(json.dumps({'mode':args.mode,'checks':receipt['summary']['checks'],
            'scientific_gate_passed':receipt['scientific_gate_passed'],'out':str(args.out)},indent=2))
        return 0
    except (ValueError,KeyError,TypeError,OSError,subprocess.CalledProcessError) as exc:
        if 'receipt' in locals():
            receipt['execution_error']=str(exc);receipt['complete']=False
            try:write_json(args.out,receipt)
            except (ValueError,OSError):pass
        print(f'Reward experiment error: {exc}',flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
