"""Supervised material-memory experiment and explicit conventional controls."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from adaptive_medium import (make_grid, initial_material, project_budget, rollout,
                             loss_gradient, motor_response_map)
from adaptive_medium_task import (TaskConfig, sample_cue, make_drive,
                                  episode_from_energies)


@dataclass
class MaterialRun:
    initial: np.ndarray
    final: np.ndarray
    curve: list
    costs: dict


def training_examples(seed,updates,cfg):
    rng=np.random.default_rng(np.random.SeedSequence([seed,10]))
    for _ in range(updates):
        yield [sample_cue(cue,rng,cfg,evaluation=False) for cue in (0,1)]


def check_training(mapping,updates,rate):
    if tuple(mapping) not in ((0,1),(1,0)) or not isinstance(updates,int) or updates<0 or not np.isfinite(rate) or rate<=0:
        raise ValueError('invalid mapping, update count or learning rate')


def train_material(grid,seed,mapping,cfg,*,updates=600,rate=.05,mode='adjoint'):
    check_training(mapping,updates,rate)
    if mode not in ('adjoint','traffic'): raise ValueError('unknown material writer')
    start=time.perf_counter(); initial=initial_material(grid,seed,mean=cfg.mean,lower=cfg.lower,upper=cfg.upper)
    g=initial.copy(); curve=[]; digest=hashlib.sha256(); a,b=grid.edges.T
    for step,sources in enumerate(training_examples(seed,updates,cfg)):
        gradients=[]; losses=[]
        for cue,source in enumerate(sources):
            digest.update(source.tobytes()); drive=make_drive(source,cfg)
            if mode=='adjoint':
                loss,ep,gradient=loss_gradient(grid,g,drive,cfg,1-2*mapping[cue])
            else:
                states=rollout(grid,g,drive,dt=cfg.dt,leak=cfg.leak)
                from adaptive_medium_task import read_motor
                ep=read_motor(states,cfg)
                gradient=-np.mean((states[:-1,a]-states[:-1,b])**2,axis=0)
                loss=.5*(ep.score-cfg.target_magnitude*(1-2*mapping[cue]))**2
            gradients.append(gradient); losses.append(loss)
        g=project_budget(g-rate*np.mean(gradients,axis=0),len(g)*cfg.mean,cfg.lower,cfg.upper)
        if step%10==0 or step==updates-1:
            curve.append({'step':step+1,'loss':float(np.mean(losses))})
    costs={'seconds':time.perf_counter()-start,'input_digest':digest.hexdigest(),
           'forward_rollouts':2*updates,'adjoint_rollouts':2*updates if mode=='adjoint' else 0,
           'solver_steps':cfg.steps*2*updates*(2 if mode=='adjoint' else 1),
           'persistent_scalars':len(g),'persistent_bytes':g.nbytes,
           'peak_trajectory_bytes':(2 if mode=='adjoint' else 1)*(cfg.steps+1)*grid.nodes*8,
           'trajectory_storage_scope':'forward/adjoint arrays only; excludes gradient temporaries'}
    return MaterialRun(initial,g,curve,costs)


def sensory_features(sources,cfg):
    x=np.asarray(sources).reshape(-1,cfg.size,cfg.size)
    return np.column_stack([x[:,1:4,:3].sum(axis=(1,2)),x[:,5:8,:3].sum(axis=(1,2))])


def train_linear(seed,mapping,cfg,*,updates=600,rate=.05):
    check_training(mapping,updates,rate)
    start=time.perf_counter(); w=np.zeros((2,2)); bias=np.zeros(2); digest=hashlib.sha256()
    for sources in training_examples(seed,updates,cfg):
        for source in sources: digest.update(source.tobytes())
        x=sensory_features(sources,cfg); logits=x@w.T+bias
        p=np.exp(logits-logits.max(axis=1,keepdims=True)); p/=p.sum(axis=1,keepdims=True)
        p[np.arange(2),mapping]-=1; p/=2
        w-=rate*(p.T@x); bias-=rate*p.sum(axis=0)
    return w,bias,{'seconds':time.perf_counter()-start,'input_digest':digest.hexdigest(),
        'persistent_scalars':6,'persistent_bytes':w.nbytes+bias.nbytes,'minibatch_updates':updates,
        'solver_steps':0,'forward_rollouts':0,'adjoint_rollouts':0}


def make_evaluation_sources(seed,cfg,per_cue=100):
    if not isinstance(per_cue,int) or per_cue<1: raise ValueError('positive example count required')
    rng=np.random.default_rng(np.random.SeedSequence([seed,20]))
    cues=np.repeat(np.arange(2),per_cue)
    return np.stack([sample_cue(int(c),rng,cfg,evaluation=True) for c in cues]),cues


def evaluate_material(grid,g,sources,cfg):
    sources=np.asarray(sources,dtype=float)
    if sources.ndim!=2 or sources.shape[1]!=grid.nodes or not np.isfinite(sources).all() or np.any(sources<0):
        raise ValueError('invalid evaluation sources')
    responses=sources@motor_response_map(grid,g,cfg).T
    return [episode_from_energies(e,cfg) for e in responses]


def score_episodes(episodes,targets):
    targets=np.asarray(targets)
    if len(episodes)!=len(targets) or not len(episodes) or not np.isin(targets,[0,1]).all():
        raise ValueError('invalid evaluator targets')
    energy=np.stack([e.energies for e in episodes])
    return {'accuracy':float(np.mean([e.valid and e.action==int(t) for e,t in zip(episodes,targets)])),
        'count':len(episodes),'invalid':sum(not e.valid for e in episodes),
        'ties':sum(e.valid and e.action is None for e in episodes),
        'scores':[e.score for e in episodes], 'energies':[e.energies.tolist() for e in episodes],
        'energy_summary':{name:getattr(energy,name)(axis=0).tolist() for name in ('min','mean','max')},
        'minimum_energy':float(min(e.energies.sum() for e in episodes))}


def interventions(run,seed,count=20):
    if not isinstance(count,int) or count<1: raise ValueError('positive shuffle count required')
    rng=np.random.default_rng(np.random.SeedSequence([seed,30]))
    return {'learned':run.final.copy(),'frozen':run.initial.copy(),'erased':run.initial.copy(),
            'shuffled':[rng.permutation(run.final) for _ in range(count)]}


def validate_material(g,grid,cfg):
    g=np.asarray(g,dtype=float)
    budget=len(grid.edges)*cfg.mean
    if (g.shape!=(len(grid.edges),) or not np.isfinite(g).all()
            or g.min()<cfg.lower-1e-12 or g.max()>cfg.upper+1e-12
            or abs(g.sum()-budget)>1e-10*budget):
        raise ValueError('invalid material checkpoint bounds, shape or budget')
    return g


def save_material(path,g,cfg):
    grid=make_grid(cfg.size); validate_material(g,grid,cfg)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps({'size':cfg.size,'config':asdict(cfg),
        'conductances':np.asarray(g).tolist()},allow_nan=False)+'\n',encoding='utf-8')


def load_material(path,grid,cfg):
    try:
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if data['size']!=grid.size or data['config']!=asdict(cfg):
            raise ValueError('checkpoint configuration mismatch')
        return validate_material(data['conductances'],grid,cfg).copy()
    except (KeyError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('malformed material checkpoint') from exc


def git_output(*args,root=None):
    return subprocess.check_output(['git',*args],cwd=root or Path(__file__).parent,
                                   text=True,stderr=subprocess.DEVNULL).strip()


def verify_lock(path,commit,*,root=None):
    root=Path(root or Path(__file__).parent).resolve(); path=Path(path).resolve()
    try:
        relative=path.relative_to(root).as_posix()
        if not commit or git_output('rev-parse','HEAD',root=root)!=commit:
            raise ValueError('full run requires the exact current lock commit')
        if git_output('status','--porcelain','--untracked-files=no',root=root):
            raise ValueError('full run requires a clean tracked tree')
        if git_output('show',f'{commit}:{relative}',root=root)!=path.read_text().strip():
            raise ValueError('protocol differs from its committed version')
        if json.loads(path.read_text()).get('locked') is not True:
            raise ValueError('protocol is not locked')
        for name in ('adaptive_medium.py','adaptive_medium_task.py','adaptive_medium_gate.py','adaptive_medium_replay.py'):
            git_output('cat-file','-e',f'{commit}:{name}',root=root)
    except (subprocess.CalledProcessError,KeyError,json.JSONDecodeError,OSError) as exc:
        raise ValueError('missing or invalid committed protocol/code') from exc


def refinement_check(grid,g,sources,cfg):
    from dataclasses import replace
    half=replace(cfg,dt=cfg.dt/2,steps=cfg.steps*2,pulse=cfg.pulse*2,read_start=2*(cfg.read_start-1)+1)
    x=evaluate_material(grid,g,sources,cfg); y=evaluate_material(grid,g,sources,half)
    delta=max(abs(a.score-b.score) for a,b in zip(x,y))
    return {'max_score_difference':float(delta),'changed_actions':sum(a.action!=b.action for a,b in zip(x,y)),
            'passed':bool(delta<.02 and all(a.valid and b.valid for a,b in zip(x,y)))}


def evaluate_seed(seed,cfg,*,updates,per_cue,shuffle_count):
    grid=make_grid(cfg.size); sources,cues=make_evaluation_sources(seed,cfg,per_cue)
    records=[]; numerical=[]
    for mapping in ((0,1),(1,0)):
        print(f'seed {seed} mapping {mapping}: training {updates} updates',flush=True)
        run=train_material(grid,seed,mapping,cfg,updates=updates)
        traffic=train_material(grid,seed,mapping,cfg,updates=updates,mode='traffic')
        w,bias,linear_cost=train_linear(seed,mapping,cfg,updates=updates)
        targets=np.asarray(mapping)[cues]; mats=interventions(run,seed,shuffle_count)
        start=time.perf_counter(); arms={}
        for name in ('learned','frozen','erased'):
            arms[name]=score_episodes(evaluate_material(grid,mats[name],sources,cfg),targets)
        shuffles=[score_episodes(evaluate_material(grid,g,sources,cfg),targets) for g in mats['shuffled']]
        for metrics in shuffles:
            # Exact aggregates plus reproducible permutation stream; avoid 20 copies of per-case traces.
            metrics.pop('scores'); metrics.pop('energies')
        arms['shuffled']={'accuracy':float(np.mean([r['accuracy'] for r in shuffles])),
            'invalid':sum(r['invalid'] for r in shuffles),'count':sum(r['count'] for r in shuffles),
            'minimum_energy':min(r['minimum_energy'] for r in shuffles),'permutation_results':shuffles}
        arms['activity_only']=score_episodes(evaluate_material(grid,traffic.final,sources,cfg),targets)
        logits=sensory_features(sources,cfg)@w.T+bias
        prediction=np.argmax(logits,axis=1); ties=logits[:,0]==logits[:,1]
        arms['linear']={'accuracy':float(np.mean((prediction==targets)&~ties)),'invalid':0,
                        'count':len(targets),'ties':int(ties.sum())}
        arms['lookup']={'accuracy':1.,'count':len(targets),'invalid':0,'oracle_cue_identity':True,
                        'persistent_scalars':2,'persistent_bytes':16}
        evaluation_seconds=time.perf_counter()-start
        for g in (run.initial,run.final,traffic.final): validate_material(g,grid,cfg)
        probes=sources[[0,per_cue]]
        refine=[refinement_check(grid,g,probes,cfg) for g in (run.initial,run.final)]
        numerical.extend(refine)
        record={'mapping':list(mapping),'arms':arms,'initial_material':run.initial.tolist(),
            'learned_material':run.final.tolist(),'traffic_material':traffic.final.tolist(),
            'training_curve':run.curve,'training_costs':run.costs,'traffic_costs':traffic.costs,
            'linear_costs':linear_cost,'linear_parameters':{'weights':w.tolist(),'biases':bias.tolist()},
            'evaluation_costs':{'seconds':evaluation_seconds,'material_maps':4+shuffle_count,
                'map_backward_steps':(4+shuffle_count)*cfg.steps,
                'source_motor_multiply_adds':(4+shuffle_count)*len(sources)*grid.nodes*2,
                'solver':'exact fixed-material motor map, verified against full rollout'},
            'refinement':refine,'shuffle_stream':[seed,30]}
        records.append(record)
        print(f"  learned={arms['learned']['accuracy']:.3f} frozen={arms['frozen']['accuracy']:.3f} "
              f"shuffle={arms['shuffled']['accuracy']:.3f} linear={arms['linear']['accuracy']:.3f}",flush=True)
    return {'seed':seed,'mappings':records,'numerical':{'passed':all(r['passed'] for r in numerical)}}


def summarize(seed_results,*,numerical_checks):
    if not seed_results: raise ValueError('no seeds to summarize')
    names=list(seed_results[0]['mappings'][0]['arms'])
    arms={}
    for name in names:
        values=[float(np.mean([m['arms'][name]['accuracy'] for m in row['mappings']])) for row in seed_results]
        if not np.isfinite(values).all(): raise ValueError('nonfinite accuracy')
        arms[name]={'mean':float(np.mean(values)),'std':float(np.std(values)),
                    'min':min(values),'max':max(values),'per_seed':values}
    learned=arms['learned']['mean']; frozen=arms['frozen']['mean']
    gain=learned-frozen
    checks={'accuracy':learned>=.90 and arms['learned']['min']>=.80,
        'gain':gain>=.20-1e-12,'erasure':learned-arms['erased']['mean']>=gain/2-1e-12,
        'spatial_arrangement':learned-arms['shuffled']['mean']>=.10-1e-12,
        'valid_episodes':all(m['arms'][name].get('invalid',0)==0 for row in seed_results
                            for m in row['mappings'] for name in names),
        'numerical':bool(numerical_checks['passed'] and all(r['numerical']['passed'] for r in seed_results))}
    return {'arms':arms,'checks':checks,'passed':all(checks.values()),
            'failed_reasons':[k for k,v in checks.items() if not v]}


def numerical_diagnostics(cfg):
    grid=make_grid(); g=initial_material(grid,11,mean=cfg.mean,lower=cfg.lower,upper=cfg.upper)
    source=sample_cue(0,np.random.default_rng(11),cfg,evaluation=False); drive=make_drive(source,cfg)
    _,ep,gradient=loss_gradient(grid,g,drive,cfg,-1)
    errors=[]; relative=[]
    for edge in (0,30,140,230,271):
        plus,minus=g.copy(),g.copy(); plus[edge]+=1e-6; minus[edge]-=1e-6
        fd=(loss_gradient(grid,plus,drive,cfg,-1)[0]-loss_gradient(grid,minus,drive,cfg,-1)[0])/2e-6
        errors.append(abs(gradient[edge]-fd)); relative.append(abs(gradient[edge]-fd)/(abs(fd)+1e-12))
    actual=rollout(grid,g,drive,dt=cfg.dt,leak=cfg.leak)
    from adaptive_medium_task import read_motor
    map_error=float(np.max(abs(motor_response_map(grid,g,cfg)@source-read_motor(actual,cfg).energies)))
    return {'gradient_max_abs_error':max(errors),'gradient_max_relative_error':max(relative),
        'response_map_max_abs_error':map_error,
        'passed':bool(max(errors)<1e-7 and map_error<1e-12)}


def write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(data,indent=2,allow_nan=False)+'\n'
    temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(text,encoding='utf-8'); temporary.replace(path)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('development','smoke','full'),default='development')
    parser.add_argument('--protocol',type=Path,default=Path(__file__).parent/'protocols/adaptive_medium.json')
    parser.add_argument('--lock-commit')
    parser.add_argument('--out',type=Path,default=Path('results/adaptive_medium/development.json'))
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try:
        protocol=json.loads(args.protocol.read_text()); cfg=TaskConfig(**protocol['config'])
        expected_thresholds={'accuracy':.90,'minimum_seed':.80,'gain':.20,'erase_fraction':.5,'shuffle_gain':.10,'refinement':.02}
        if protocol['rate']!=.05 or protocol['thresholds']!=expected_thresholds:
            raise ValueError('rate or verdict thresholds differ from the implemented protocol')
        if protocol['development_seeds']!=[11,23,41] or protocol['heldout_seeds']!=list(range(101,111)):
            raise ValueError('seed split changed from approved protocol')
        if args.mode=='full':
            verify_lock(args.protocol,args.lock_commit)
            if protocol['heldout_seeds']!=list(range(101,111)) or protocol['development_seeds']!=[11,23,41]:
                raise ValueError('seed split changed from approved protocol')
            if (protocol['updates'],protocol['per_cue'],protocol['shuffle_count'])!=(600,100,20):
                raise ValueError('full protocol budgets differ from approved lock')
        seeds=[11] if args.mode=='smoke' else protocol['heldout_seeds' if args.mode=='full' else 'development_seeds']
        updates=5 if args.mode=='smoke' else protocol['updates']
        per_cue=3 if args.mode=='smoke' else protocol['per_cue']
        shuffles=2 if args.mode=='smoke' else protocol['shuffle_count']
        receipt={'protocol_version':protocol['version'],'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            'code_commit':git_output('rev-parse','HEAD'),'mode':args.mode,'held_out':args.mode=='full',
            'working_tree_dirty':bool(git_output('status','--porcelain','--untracked-files=no')),
            'config':asdict(cfg),'updates':updates,'per_cue':per_cue,'shuffle_count':shuffles,
            'numerical_checks':numerical_diagnostics(cfg),'seed_results':[],'complete':False}
        if args.resume:
            if args.mode!='full': raise ValueError('resume is only supported for a locked full run')
            old=json.loads(args.out.read_text())
            for key in ('protocol_sha256','code_commit','mode'):
                if old[key]!=receipt[key]: raise ValueError('resume provenance mismatch')
            done=[x['seed'] for x in old['seed_results']]
            if done!=seeds[:len(done)]: raise ValueError('resume seed prefix mismatch')
            receipt['seed_results']=old['seed_results']
        start=time.perf_counter()
        for seed in seeds[len(receipt['seed_results']):]:
            receipt['seed_results'].append(evaluate_seed(seed,cfg,updates=updates,per_cue=per_cue,shuffle_count=shuffles))
            write_json(args.out,receipt)
        receipt['summary']=summarize(receipt['seed_results'],numerical_checks=receipt['numerical_checks'])
        receipt['complete']=True; receipt['run_seconds']=time.perf_counter()-start
        receipt['scientific_gate_passed']=bool(args.mode=='full' and receipt['summary']['passed'])
        write_json(args.out,receipt)
        if args.replay:
            from adaptive_medium_replay import write_replay, replay_episodes
            write_replay(args.replay,grid=make_grid(),receipt=receipt,episodes=replay_episodes(receipt))
        print(json.dumps({'mode':args.mode,'criteria':receipt['summary']['checks'],
            'scientific_gate_passed':receipt['scientific_gate_passed'],'saved':str(args.out)},indent=2))
        return 0
    except (ValueError,KeyError,TypeError,OSError,subprocess.CalledProcessError) as exc:
        if 'receipt' in locals():
            receipt['complete']=False; receipt['execution_error']=str(exc)
            try: write_json(args.out,receipt)
            except (ValueError,OSError): pass
        print(f'Experiment error: {exc}',flush=True)
        return 2


if __name__=='__main__': raise SystemExit(main())
