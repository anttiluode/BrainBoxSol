"""History -> material -> temporal port -> frozen receiver; locked fresh gate."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from adaptive_medium import make_grid,initial_material
from adaptive_medium_task import TaskConfig
from adaptive_medium_gate import verify_lock,git_output,write_json
from history_waveform import (make_dataset,match_waveform,probe_waveform,
                              receive_waveforms,fit_threshold,predict_threshold)

THRESHOLDS={'accuracy':.8,'gain':.2,'control_ceiling':.6,'invariant_tolerance':1e-10,'refinement':.02}


def default_protocol():
    return {'version':'history-waveform-v1','locked':False,'config':asdict(TaskConfig()),
        'history_pulses':3,'write_rate':.005,'noise':.02,'std':.15,
        'train_pairs':32,'test_pairs':64,'development_seeds':[401,409,419],
        'heldout_seeds':list(range(501,511)),'thresholds':THRESHOLDS.copy()}


def validate_protocol(p):
    try:
        if (p['config']!=asdict(TaskConfig()) or p['development_seeds']!=[401,409,419]
                or p['heldout_seeds']!=list(range(501,511)) or p['thresholds']!=THRESHOLDS):
            raise ValueError('fixed configuration, seed split, or thresholds changed')
        for k in ('history_pulses','train_pairs','test_pairs'):
            if not isinstance(p[k],int) or isinstance(p[k],bool) or p[k]<1:raise ValueError('positive integer count required')
        if not np.isfinite([p['write_rate'],p['noise'],p['std']]).all() or min(p['write_rate'],p['std'])<=0 or p['noise']<0:
            raise ValueError('invalid waveform parameters')
    except (KeyError,TypeError) as exc:raise ValueError('incomplete waveform protocol') from exc


def summarize(rows):
    if not rows:raise ValueError('no seed results')
    names=rows[0]['arms'];arms={}
    for name in names:
        a=[row['arms'][name]['accuracy'] for row in rows]
        if not np.isfinite(a).all() or min(a)<0 or max(a)>1:raise ValueError('invalid accuracy')
        arms[name]={'mean':float(np.mean(a)),'std':float(np.std(a)),'min':min(a),'max':max(a),'per_seed':a}
    r=arms['joint']['mean'];checks={'accuracy':r>=THRESHOLDS['accuracy']}
    for name in ('fixed','shuffled','erased'):
        checks[name+'_advantage']=r-arms[name]['mean']>=THRESHOLDS['gain']-1e-12
        checks[name+'_ceiling']=arms[name]['mean']<=THRESHOLDS['control_ceiling']
    checks['invariants']=all(row['invariants']['passed'] for row in rows)
    checks['refinement']=all(row['refinement']['passed'] for row in rows)
    return {'arms':arms,'checks':checks,'passed':all(checks.values()),
            'failed_reasons':[k for k,v in checks.items() if not v]}


def evaluate_seed(seed,p):
    cfg=TaskConfig(**p['config']);grid=make_grid();start=time.perf_counter()
    kwargs={k:p[k] for k in ('history_pulses','write_rate','noise')}
    tr=make_dataset(seed,p['train_pairs'],cfg,split=1,**kwargs)
    te=make_dataset(seed,p['test_pairs'],cfg,split=2,**kwargs)
    receiver=initial_material(grid,100000+seed)
    match=lambda y:np.stack([match_waveform(x,cfg.dt,std=p['std']) for x in y])
    a=match(tr['raw']);b=match(te['raw'])
    reference=match_waveform(tr['raw'].mean(axis=0),cfg.dt,std=p['std'])
    permutation_rng=np.random.default_rng(np.random.SeedSequence([seed,94]))
    perm_train=permutation_rng.permutation(len(a));perm_test=permutation_rng.permutation(len(b))
    target_energy=cfg.steps*cfg.dt*(1+p['std']**2)
    energy_only=lambda y:y*np.sqrt(target_energy/(cfg.dt*np.sum(y*y,axis=1)))[:,None]
    # Labels never enter these waveforms or the receiver dynamics.
    inputs={'joint':(a,b),'raw':(tr['raw'],te['raw']),
        'energy_only':(energy_only(tr['raw']),energy_only(te['raw'])),
        'fixed':(np.tile(reference,(len(a),1)),np.tile(reference,(len(b),1))),
        'shuffled':(a[perm_train],b[perm_test]),'erased':(match(tr['erased']),match(te['erased']))}
    reference_charge=cfg.dt*reference.sum()
    inputs['charge_only']=(np.sum(tr['raw'],axis=1)[:,None]*cfg.dt/reference_charge*reference,
                           np.sum(te['raw'],axis=1)[:,None]*cfg.dt/reference_charge*reference)
    arms={}
    for name,(x,y) in inputs.items():
        train_voltage=receive_waveforms(grid,receiver,x,cfg);test_voltage=receive_waveforms(grid,receiver,y,cfg)
        model=fit_threshold(train_voltage,tr['labels']);pred=predict_threshold(test_voltage,model)
        arms[name]={'accuracy':float(np.mean(pred==te['labels'])),'count':len(pred),
            'model':model,'train_voltage':train_voltage.tolist(),'test_voltage':test_voltage.tolist(),
            'predictions':pred.tolist(),'class_means':[float(test_voltage[te['labels']==k].mean()) for k in (0,1)]}
    # A simple scalar peak is also permitted to read the conditioned waveform.
    peak_model=fit_threshold(a.max(axis=1),tr['labels']);peak_pred=predict_threshold(b.max(axis=1),peak_model)
    arms['peak_scalar']={'accuracy':float(np.mean(peak_pred==te['labels'])),'count':len(peak_pred),'model':peak_model}
    all_matched=np.concatenate([x for name,xy in inputs.items() if name in ('joint','fixed','shuffled','erased') for x in xy])
    q=cfg.dt*all_matched.sum(axis=1);e=cfg.dt*np.sum(all_matched**2,axis=1)
    qerr=float(np.max(abs(q-cfg.steps*cfg.dt)));eerr=float(np.max(abs(e-target_energy)))
    invariants={'max_charge_error':qerr,'max_energy_error':eerr,'common_charge':cfg.steps*cfg.dt,
        'common_energy':target_energy,'minimum_current':float(all_matched.min()),
        'samples':cfg.steps,'onset':0.,'duration':cfg.steps*cfg.dt,
        'passed':bool(max(qerr,eerr)<THRESHOLDS['invariant_tolerance'] and all_matched.min()>0)}
    # Fixed learned fields, noiseless first pair: halve both boxes' solver step.
    half=replace(cfg,dt=cfg.dt/2,steps=2*cfg.steps,pulse=2*cfg.pulse,read_start=2*(cfg.read_start-1)+1)
    coarse=[];fine=[]
    for ex in te['examples']:
        material=np.asarray(ex['material'])
        y=match_waveform(np.asarray(ex['raw_trace']),cfg.dt,std=p['std'])
        yy=match_waveform(probe_waveform(grid,material,np.repeat(te['probe_drive'],2,axis=0),half),half.dt,std=p['std'])
        coarse.append(float(receive_waveforms(grid,receiver,y[None,:],cfg)[0]))
        fine.append(float(receive_waveforms(grid,receiver,yy[None,:],half)[0]))
    delta=float(np.max(abs(np.asarray(coarse)-fine)))
    result={'seed':seed,'arms':arms,'labels_train':tr['labels'].tolist(),'labels_test':te['labels'].tolist(),
        'receiver_material':receiver.tolist(),'initial_material':te['initial_material'],'examples':te['examples'],
        'primary_waveforms':b.tolist(),'fixed_reference':reference.tolist(),
        'shuffled_test_indices':perm_test.tolist(),'shuffled_train_indices':perm_train.tolist(),
        'input_digests':{'train':tr['history_digest'],'test':te['history_digest']},'invariants':invariants,
        'refinement':{'coarse_voltage':coarse,'half_step_voltage':fine,'max_voltage_difference':delta,
                      'passed':delta<THRESHOLDS['refinement']},
        'memory':{**te['memory'],'receiver_persistent_scalars':len(receiver),'receiver_persistent_bytes':receiver.nbytes,
            'readout_parameters_per_arm':2,'readout_payload_bytes_per_arm':16,
            'receiver_fast_state_scalars':grid.nodes,'box1_fast_state_scalars':grid.nodes},
        'costs':{'seconds':time.perf_counter()-start,
            'history_forward_rollouts':(len(a)+len(b))*p['history_pulses'],
            'probe_forward_rollouts':len(a)+len(b)+4,
            'receiver_forward_rollouts':len(inputs)*(len(a)+len(b))+4,
            'adjoint_rollouts':0,'readout_training_examples_per_arm':len(a)}}
    print(f'seed {seed}: '+str({name:round(v['accuracy'],4) for name,v in arms.items()})+f' refinement={delta:.4g}',flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','development','full'),default='development')
    parser.add_argument('--protocol',type=Path,default=Path(__file__).parent/'protocols/history_waveform.json')
    parser.add_argument('--lock-commit');parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    args.out=args.out or Path('results/history_waveform')/f'{args.mode}.json'
    try:
        p=json.loads(args.protocol.read_text());validate_protocol(p)
        if args.mode=='full':
            verify_lock(args.protocol,args.lock_commit)
            for name in ('history_waveform.py','waveform_gate.py'):
                git_output('cat-file','-e',f'{args.lock_commit}:{name}')
        actual=p.copy()
        if args.mode=='smoke':actual.update(train_pairs=2,test_pairs=2,history_pulses=1)
        seeds=[401] if args.mode=='smoke' else p['heldout_seeds' if args.mode=='full' else 'development_seeds']
        receipt={'mode':args.mode,'protocol':p,'actual':actual,'code_commit':git_output('rev-parse','HEAD'),
            'tracked_tree_dirty':bool(git_output('status','--porcelain','--untracked-files=no')),
            'source_hashes':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                             for name in ('history_waveform.py','waveform_gate.py')},
            'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            'held_out':args.mode=='full','seed_results':[],'complete':False}
        for seed in seeds:
            receipt['seed_results'].append(evaluate_seed(seed,actual));write_json(args.out,receipt)
        receipt['summary']=summarize(receipt['seed_results']);receipt['complete']=True
        receipt['scientific_gate_passed']=bool(receipt['held_out'] and receipt['summary']['passed'])
        write_json(args.out,receipt);print(json.dumps(receipt['summary']['checks'],indent=2));return 0
    except (ValueError,KeyError,TypeError,OSError,subprocess.CalledProcessError) as exc:
        if 'receipt' in locals():
            receipt['complete']=False;receipt['execution_error']=str(exc)
            try:write_json(args.out,receipt)
            except (ValueError,OSError):pass
        print(f'Waveform experiment error: {exc}',flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
