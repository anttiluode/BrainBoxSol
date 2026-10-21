"""Paired, source-locked local mobility memory experiment; NumPy CPU only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import numpy as np
from adaptive_medium import make_grid,operator
from local_slowness import evolve,wait_material,probe,steps_for

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'protocols/local_slowness.json'
SOURCE_FILES=('local_slowness.py','local_slowness_gate.py','adaptive_medium.py','protocols/local_slowness.json')
WRITERS=('local','trace','noise_off','noise_only')
ARMS=('local','trace','uniform','shuffled','noise_off','noise_only')


def load_protocol():return json.loads(PROTOCOL.read_text())


def rng(seed,split,stream):
    return np.random.default_rng(np.random.SeedSequence([20261021,int(seed),int(split),int(stream)]))


def effective(protocol,mode):
    if mode not in ('smoke','development','full'):raise ValueError('unknown mode')
    p=copy.deepcopy(protocol)
    if mode=='smoke':
        p['evaluation'].update(training_pairs=2,test_pairs=3,retention_delays=[0.,2.],
            rewrite_gap=2.,rewrite_delays=[0.],primary_delay=2.)
    return p


def drives(seed,split,pairs,cfg,*,tag,labels=None,noise=True,cue=True):
    n=cfg['size']**2;steps=steps_for(cfg['write_duration'],cfg['dt'])
    hold=steps_for(cfg['noise_hold'],cfg['dt']);pulse=steps_for(cfg['cue_duration'],cfg['dt'])
    labels=np.tile(np.arange(3),pairs) if labels is None else np.asarray(labels)
    if labels.shape!=(pairs*3,) or not np.isin(labels,[0,1,2]).all():raise ValueError('invalid cue classes')
    generator=rng(seed,split,tag)
    background=generator.normal(0,cfg['noise_sd'],(pairs,int(np.ceil(steps/hold)),n))
    background=np.repeat(np.repeat(background,hold,axis=1)[:,:steps],3,axis=0)
    if not noise:background.fill(0)
    amplitudes=np.repeat(generator.uniform(.8,1.2,pairs),3)
    if cue:
        locations=np.array([0,cfg['size']-1,n-cfg['size']])
        background[np.arange(3*pairs)[:,None],np.arange(pulse)[None,:],locations[labels,None]]+=amplitudes[:,None]
    return background


def fit_decoder(features,labels,noise_floor):
    x=np.asarray(features,dtype=float);y=np.asarray(labels)
    if x.ndim!=2 or y.shape!=(len(x),) or set(y)!={0,1,2} or not np.isfinite(x).all() or noise_floor<=0:
        raise ValueError('three finite calibration classes required')
    scale=np.maximum(np.std(x,axis=0),noise_floor)
    return {'scale':scale.tolist(),'centers':[x[y==k].mean(axis=0).tolist() for k in range(3)]}


def predict(features,model):
    x=np.asarray(features);centers=np.asarray(model['centers']);scale=np.asarray(model['scale'])
    return np.argmin(np.sum(((x[:,None,:]-centers[None,:,:])/scale)**2,axis=2),axis=1)


def _add_cost(cost,addition):
    for k,v in addition.items():cost[k]=cost.get(k,0)+int(v)


def write_all(A,initial,seed,split,pairs,cfg,*,tag,labels=None):
    states={};costs={}
    for writer in WRITERS:
        drive=drives(seed,split,pairs,cfg,tag=tag,labels=labels,
            noise=writer!='noise_off',cue=writer!='noise_only')
        start=initial[writer] if isinstance(initial,dict) else initial
        states[writer],_,costs[writer]=evolve(A,start,drive,dt=cfg['dt'],rate=cfg['write_rate'],
            local=writer!='trace',floor=cfg['activity_floor'])
    return states,costs


def wait_all(states,duration,cfg,costs):
    out={}
    for writer,m in states.items():
        out[writer],count=wait_material(m,duration,dt=cfg['dt'],rate=cfg['rest_rate'],local=writer!='trace')
        _add_cost(costs[writer],{'material_vector_steps':count})
    return out


def observe(A,states,seed,split,pairs,cfg,*,condition):
    n=cfg['size']**2;ports=[0,cfg['size']-1,n-cfg['size']]
    noise=np.repeat(rng(seed,split,300+condition).normal(0,cfg['observation_noise_sd'],
        (pairs,3*len(cfg['sample_times']))),3,axis=0)
    features={};examples={};read_costs={};budget_error=0.
    permutations=rng(seed,split,200).random((pairs*3,n)).argsort(axis=1)
    for arm in ARMS:
        m=states['local'] if arm in ('uniform','shuffled') else states[arm]
        if arm=='uniform':m=np.full_like(m,cfg['initial_mass_mean'])
        if arm=='shuffled':m=np.take_along_axis(m,permutations,axis=1)
        features[arm],read_costs[arm]=probe(A,m,ports,cfg)
        features[arm]+=noise
        examples[arm]=m[:3].tolist()
        budget_error=max(budget_error,float(np.max(np.abs(m.mean(axis=1)-cfg['initial_mass_mean']))))
    return features,examples,read_costs,budget_error


def refinement(A,seed,cfg,ecfg):
    # Actual first paired triplets, with the same amplitudes/noise as full batches.
    train=drives(seed,1,ecfg['training_pairs'],cfg,tag=1)[:3]
    test=drives(seed,2,ecfg['test_pairs'],cfg,tag=1)[:3]
    offset=np.repeat(rng(seed,2,41).integers(1,3,ecfg['test_pairs']),3)
    replacement=(np.tile(np.arange(3),ecfg['test_pairs'])+offset)%3
    rewrite=drives(seed,2,ecfg['test_pairs'],cfg,tag=2,labels=replacement)[:3]
    ports=[0,cfg['size']-1,cfg['size']**2-cfg['size']];by_writer={}
    for writer in ('local','trace'):
        levels=[]
        for multiplier in (1,2):
            c=cfg|{'dt':cfg['dt']/multiplier};initial=np.full((3,len(A)),cfg['initial_mass_mean'])
            def write(start,drive):
                return evolve(A,start,np.repeat(drive,multiplier,axis=1),dt=c['dt'],rate=c['write_rate'],local=writer=='local',floor=c['activity_floor'])[0]
            def observe(m):return probe(A,m,ports,c)[0]
            def wait(m,duration):return wait_material(m,duration,dt=c['dt'],rate=c['rest_rate'],local=writer=='local')[0]
            training=write(initial,train);written=write(initial,test)
            retained=wait(written,ecfg['primary_delay'])
            replaced=write(wait(written,ecfg['rewrite_gap']),rewrite)
            later=wait(replaced,max(ecfg['rewrite_delays']))
            levels.append({'after_write':(np.concatenate([training,written]),np.concatenate([observe(training),observe(written)])),
                'after_wait':(retained,observe(retained)),
                'after_rewrite':(np.concatenate([replaced,later]),np.concatenate([observe(replaced),observe(later)]))})
        by_writer[writer]={phase:{'material_max_abs':float(np.max(np.abs(levels[0][phase][0]-levels[1][phase][0]))),
            'probe_voltage_max_abs':float(np.max(np.abs(levels[0][phase][1]-levels[1][phase][1])))} for phase in levels[0]}
    rows=[v for phases in by_writer.values() for v in phases.values()]
    return {'sampled_histories_per_phase':3,'by_writer':by_writer,
        'training_drive_sha256':hashlib.sha256(train.tobytes()).hexdigest(),
        'material_max_abs':max(v['material_max_abs'] for v in rows),
        'probe_voltage_max_abs':max(v['probe_voltage_max_abs'] for v in rows),
        'scope':'First actual training/test triplets, both primary writers, writing, primary waiting, replacement and later waiting; freshly rewritten at each dt. Does not check every trial.'}


def run_device(seed,p):
    cfg=p['model'];ecfg=p['evaluation'];n=cfg['size']**2;grid=make_grid(cfg['size'])
    g=cfg['conductance_mean']*rng(seed,0,1).uniform(.85,1.15,len(grid.edges))
    A=operator(grid,g,leak=cfg['leak']);train_pairs=ecfg['training_pairs'];test_pairs=ecfg['test_pairs']
    training_y=np.tile(np.arange(3),train_pairs);y=np.tile(np.arange(3),test_pairs)
    initial=lambda pairs:np.full((3*pairs,n),cfg['initial_mass_mean'])
    trained,train_cost=write_all(A,initial(train_pairs),seed,1,train_pairs,cfg,tag=1)
    train_x,_,train_reads,budget_error=observe(A,trained,seed,1,train_pairs,cfg,condition=0)
    models={arm:fit_decoder(train_x[arm],training_y,cfg['observation_noise_sd']) for arm in ARMS}
    test,write_cost=write_all(A,initial(test_pairs),seed,2,test_pairs,cfg,tag=1)
    costs=copy.deepcopy(train_cost)
    for w in WRITERS:_add_cost(costs[w],write_cost[w])
    read_costs={arm:train_reads[arm] for arm in ARMS}
    conditions={};examples={}
    def record(name,states,target,ci,old=None):
        nonlocal budget_error
        x,masses,reads,error=observe(A,states,seed,2,test_pairs,cfg,condition=ci)
        budget_error=max(budget_error,error);conditions[name]={};examples[name]=masses
        for arm in ARMS:
            decisions=predict(x[arm],models[arm]);read_costs[arm]+=reads[arm]
            row={'accuracy':float(np.mean(decisions==target)),'predictions':decisions.tolist(),
                 'target_classes':target.tolist(),
                 'feature_class_means':[x[arm][target==k].mean(axis=0).tolist() for k in range(3)],
                 'first_triplet_features':x[arm][:3].tolist()}
            if old is not None:row['old_class_accuracy']=float(np.mean(decisions==old))
            conditions[name][arm]=row
    for ci,delay in enumerate(ecfg['retention_delays']):
        states=wait_all(test,delay,cfg,costs)
        record(f'retention:{delay:g}',states,y,ci+1)
    pre_rewrite=wait_all(test,ecfg['rewrite_gap'],cfg,costs)
    offset=np.repeat(rng(seed,2,41).integers(1,3,test_pairs),3)
    replacement=(y+offset)%3
    rewritten,rewrite_cost=write_all(A,pre_rewrite,seed,2,test_pairs,cfg,tag=2,labels=replacement)
    for w in WRITERS:_add_cost(costs[w],rewrite_cost[w])
    for ci,delay in enumerate(ecfg['rewrite_delays']):
        states=wait_all(rewritten,delay,cfg,costs)
        record(f'rewrite:{delay:g}',states,replacement,100+ci,old=y)
    return {'seed':seed,'conditions':conditions,'calibration':models,'example_materials':examples,
        'fixed_conductances':g.tolist(),'costs':costs,'probe_vector_steps':read_costs,
        'max_material_budget_error':budget_error,'refinement':refinement(A,seed,cfg,ecfg),
        'memory':{'persistent_material_scalars_per_arm':n,'persistent_material_bytes_per_arm':8*n,
            'fixed_conductance_scalars':len(g),'fast_voltage_scalars_per_trial':n,
            'dense_fixed_operator_cache_scalars':n*n,
            'decoder_scalars_per_arm':12*len(cfg['sample_times']),
            'observed_voltage_scalars_per_probe':3*len(cfg['sample_times']),
            'fast_activity_at_every_probe':'zero','read_plasticity':'frozen',
            'uniform_shuffled_writer':'share local-written field; independent read controls',
            'evaluation_storage':'Offline paired trials and calibration arrays are external to material; nine-sample buffer per probe.'}}


def summarize(devices,p,held_out):
    means={condition:{arm:float(np.mean([d['conditions'][condition][arm]['accuracy'] for d in devices]))
        for arm in ARMS} for condition in devices[0]['conditions']}
    ecfg=p['evaluation'];rules=p['claims'];primary=f"retention:{ecfg['primary_delay']:g}";rewrite=f"rewrite:{ecfg['primary_rewrite_delay']:g}"
    delayed=np.array([d['conditions'][primary]['local']['accuracy']-d['conditions'][primary]['trace']['accuracy'] for d in devices])
    initial=np.array([d['conditions']['retention:0']['local']['accuracy']-d['conditions']['retention:0']['trace']['accuracy'] for d in devices])
    gain=delayed-initial
    cost=means[rewrite]['trace']-means[rewrite]['local']
    intervals=np.quantile(gain[rng(0,0,99).integers(0,len(gain),(2000,len(gain)))].mean(axis=1),[.025,.975])
    numerical=all(d['refinement']['material_max_abs']<=ecfg['refinement_mass_max_abs']
        and d['refinement']['probe_voltage_max_abs']<=ecfg['refinement_voltage_max_abs'] for d in devices)
    mechanism={'initial_accuracy':means['retention:0']['local']>=rules['initial_accuracy_min'],
        'controls':all(means['retention:0'][a]<=rules['control_accuracy_max'] for a in ('uniform','shuffled','noise_only')),
        'numerical_refinement':numerical}
    criteria={'initial_accuracy':mechanism['initial_accuracy'] and means['retention:0']['trace']>=rules['initial_accuracy_min'],
        'retention_gain':float(gain.mean())>=rules['retention_gain_min'],
        'device_consistency':float(np.mean(gain>0))>=rules['positive_device_fraction_min'],
        'rewrite_cost':cost<=rules['rewrite_accuracy_cost_max'],
        'rewrite_quality':means[rewrite]['local']>=rules['rewrite_accuracy_min'],'numerical_refinement':numerical}
    return {'mean_accuracy':means,'retention_gain_local_minus_trace':float(gain.mean()),
        'per_device_retention_gain':gain.tolist(),'positive_device_fraction':float(np.mean(gain>0)),
        'delayed_accuracy_gain_local_minus_trace':float(delayed.mean()),
        'retention_gain_definition':rules['retention_gain_definition'],
        'paired_device_bootstrap_95':intervals.tolist(),'rewrite_accuracy_cost':cost,
        'gates':{'material_memory':{'criteria':mechanism,'passed':bool(all(mechanism.values())) if held_out else None},
            'retention_without_rewrite_cost':{'criteria':criteria,'passed':bool(all(criteria.values())) if held_out else None}},
        'replication_unit':'device; each accuracy averages histories within a device'}


def source_hashes():return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCE_FILES}


def check_lock(lock,p):
    if not isinstance(lock,str) or not re.fullmatch('[0-9a-f]{40}',lock) or not p['locked']:
        raise ValueError('full run requires an exact clean source/protocol lock')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    if head!=lock or dirty:raise ValueError('full run requires a clean checkout of the lock')
    expected={f:hashlib.sha256(subprocess.check_output(['git','show',lock+':'+f],cwd=ROOT)).hexdigest() for f in SOURCE_FILES}
    if source_hashes()!=expected:raise ValueError('source differs from lock')


def run_experiment(mode='development',lock=None):
    start_hashes=source_hashes()
    protocol=load_protocol()
    if mode=='full':check_lock(lock,protocol)
    p=effective(protocol,mode)
    devices=[run_device(s,p) for s in p['seeds'][mode]]
    if source_hashes()!=start_hashes:raise ValueError('source changed during execution')
    return {'experiment':'local_slowness','mode':mode,'held_out':mode=='full','lock_commit':lock,
        'protocol':protocol,'effective_config':p,'source_sha256':start_hashes,'devices':devices,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
            'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}},
        'summary':summarize(devices,p,mode=='full')}


def close(saved,expected,label='receipt'):
    if isinstance(expected,dict):
        if not isinstance(saved,dict) or set(saved)!=set(expected):raise ValueError('keys differ: '+label)
        for k in expected:close(saved[k],expected[k],label+'.'+k)
    elif isinstance(expected,list):
        if not isinstance(saved,list) or len(saved)!=len(expected):raise ValueError('length differs: '+label)
        for i,(a,b) in enumerate(zip(saved,expected)):close(a,b,label+f'[{i}]')
    elif isinstance(expected,float):
        if not np.isfinite(saved) or not np.isclose(saved,expected,rtol=1e-8,atol=1e-10):raise ValueError('value differs: '+label)
    elif type(saved)!=type(expected) or saved!=expected:raise ValueError('exact field differs: '+label)


def audit_receipt(receipt):
    if receipt.get('held_out') and not receipt.get('protocol',{}).get('locked'):
        raise ValueError('full replay requires a locked protocol')
    if receipt['protocol']!=load_protocol() or receipt['source_sha256']!=source_hashes():raise ValueError('audit source/protocol mismatch')
    if receipt['mode'] not in ('smoke','development','full') or receipt['held_out']!=(receipt['mode']=='full'):raise ValueError('invalid mode flags')
    if receipt['held_out']:
        lock=receipt['lock_commit']
        if not isinstance(lock,str) or not re.fullmatch('[0-9a-f]{40}',lock):raise ValueError('invalid recorded lock')
        expected={f:hashlib.sha256(subprocess.check_output(['git','show',lock+':'+f],cwd=ROOT)).hexdigest() for f in SOURCE_FILES}
        if expected!=receipt['source_sha256']:raise ValueError('recorded source lock differs')
    p=effective(receipt['protocol'],receipt['mode'])
    close(receipt['effective_config'],p)
    if [d['seed'] for d in receipt['devices']]!=p['seeds'][receipt['mode']]:raise ValueError('seed mismatch')
    expected=[run_device(s,p) for s in p['seeds'][receipt['mode']]]
    close(receipt['devices'],expected)
    close(receipt['summary'],summarize(expected,p,receipt['held_out']))
    return {'passed':True,'devices_replayed':len(expected),'scope':'Entire writer, waiting, replacement, calibration and all readouts regenerated from seeds; all saved fields/decisions/costs/summary checked.',
            'float_tolerance':{'rtol':1e-8,'atol':1e-10},'source_lock_checked':receipt['held_out']}


def save_receipt(result,path):
    # Separate device files keep raw receipts reviewable and simple to retrieve.
    if path.exists():raise ValueError('refusing to overwrite a result')
    path.parent.mkdir(parents=True,exist_ok=True);manifest=copy.deepcopy(result);manifest.pop('devices')
    manifest['device_receipts']=[]
    for d in result['devices']:
        dest=path.with_name(path.stem+f"-{d['seed']}.json")
        content=json.dumps(d,separators=(',',':'),allow_nan=False)+'\n'
        with dest.open('x') as stream:stream.write(content)
        manifest['device_receipts'].append({'path':dest.name,'sha256':hashlib.sha256(content.encode()).hexdigest()})
    with path.open('x') as stream:json.dump(manifest,stream,indent=2,allow_nan=False);stream.write('\n')


def load_receipt(path):
    r=json.loads(path.read_text());r['devices']=[]
    for f in r.pop('device_receipts'):
        content=(path.parent/f['path']).read_bytes()
        if hashlib.sha256(content).hexdigest()!=f['sha256']:raise ValueError('device file hash mismatch')
        r['devices'].append(json.loads(content))
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['smoke','development','full'],default='development')
    parser.add_argument('--lock-commit');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--audit',type=Path)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('refusing to overwrite output')
    if args.audit:
        result=audit_receipt(load_receipt(args.audit))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    else:
        result=run_experiment(args.mode,args.lock_commit)
        save_receipt(result,args.out)
        # Preserve first outcomes before expensive replay; write a separate audit.
        audit_path=args.out.with_name(args.out.stem+'-audit.json')
        try:audit=audit_receipt(result)
        except Exception as error:
            with audit_path.open('x') as stream:json.dump({'passed':False,'error':str(error)},stream)
            raise
        with audit_path.open('x') as stream:json.dump(audit,stream,indent=2);stream.write('\n')
    print(json.dumps(result.get('summary',result),indent=2))


if __name__=='__main__':main()
