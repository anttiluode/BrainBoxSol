"""Transmit a history-dependent RC response through one scalar temporal port.

The material writer receives activity only. A receiver receives current samples
only. Label-dependent calibration happens outside both physical media.
"""
from __future__ import annotations
import numpy as np
from adaptive_medium import rollout,transition,project_budget
from adaptive_medium_gate import validate_material


def match_waveform(raw,dt,*,mean=1.,std=.15):
    """Offline conditioner: common charge and squared-current integral.

Adds a DC pedestal and rescales the centered waveform; this is an external,
whole-window operation, not a claim of instantaneous biological normalization.
"""
    y=np.asarray(raw,dtype=float)
    if y.ndim!=1 or len(y)<2 or not np.isfinite(y).all() or not np.isfinite([dt,mean,std]).all() or min(dt,mean,std)<=0:
        raise ValueError('finite nonconstant waveform and positive conditioner parameters required')
    centered=y-y.mean();scale=np.sqrt(np.mean(centered**2))
    if scale<=1e-14:raise ValueError('constant waveform has no shape to normalize')
    result=mean+std*centered/scale
    if result.min()<=0:raise ValueError('conditioned waveform loses common positive support')
    return result


def write_history(grid,initial,drives,cfg,*,rate):
    """Activity-only traffic reinforcement; no history ID or outcome enters."""
    g=validate_material(initial,grid,cfg).copy();a,b=grid.edges.T
    if not np.isfinite(rate) or rate<=0:raise ValueError('positive write rate required')
    for drive in drives:
        states=rollout(grid,g,drive,dt=cfg.dt,leak=cfg.leak)
        traffic=np.mean((states[:-1,a]-states[:-1,b])**2,axis=0)
        if traffic.mean()<=0:raise ValueError('history must cause activity')
        g=project_budget(g+rate*(traffic/traffic.mean()-1),len(g)*cfg.mean,cfg.lower,cfg.upper)
    return validate_material(g,grid,cfg)


def probe_waveform(grid,material,drive,cfg):
    # rollout starts at zero. History activity cannot cross this boundary.
    return rollout(grid,material,drive,dt=cfg.dt,leak=cfg.leak)[1:,2*cfg.size+3].copy()


def receive_waveforms(grid,material,waveforms,cfg):
    """Frozen second grid: inject at center, observe final center voltage."""
    ys=np.asarray(waveforms,dtype=float)
    if ys.ndim!=2 or ys.shape[1]!=cfg.steps or not np.isfinite(ys).all():raise ValueError('invalid receiver waveforms')
    M=transition(grid,material,dt=cfg.dt,leak=cfg.leak);u=np.zeros((len(ys),grid.nodes));port=(grid.nodes-1)//2
    for current in ys.T:
        u=u@M.T;u[:,port]+=cfg.dt*current
    return u[:,port].copy()


def fit_threshold(values,labels):
    x=np.asarray(values,dtype=float);y=np.asarray(labels)
    if x.ndim!=1 or y.shape!=x.shape or not np.isfinite(x).all() or set(y)!={0,1}:raise ValueError('two finite training classes required')
    means=[float(x[y==k].mean()) for k in (0,1)]
    return {'threshold':sum(means)/2,'direction':1 if means[1]>=means[0] else -1}


def predict_threshold(values,model):
    return (model['direction']*(np.asarray(values)-model['threshold'])>0).astype(int)


def make_dataset(seed,pairs,cfg,*,split,history_pulses,write_rate,noise):
    """Paired spatial histories; probe, initial field and nuisance draws match.

The experiment generator chooses historical stimulation sites. Neither the
material writer nor the receiver is passed the resulting history label.
"""
    import hashlib
    from adaptive_medium import make_grid,initial_material
    from adaptive_medium_task import sample_cue,make_drive
    if (not isinstance(pairs,int) or pairs<1 or not isinstance(history_pulses,int)
            or history_pulses<1 or split not in (1,2) or not np.isfinite(noise) or noise<0):
        raise ValueError('invalid paired history dataset')
    grid=make_grid(cfg.size);initial=initial_material(grid,seed)
    source=np.zeros(grid.nodes);source[2*cfg.size+1]=1
    probe=make_drive(source,cfg);baseline=probe_waveform(grid,initial,probe,cfg)
    raw=[];erased=[];labels=[];examples=[];digest=hashlib.sha256()
    for pair in range(pairs):
        noise_rng=np.random.default_rng(np.random.SeedSequence([seed,split,pair,92]))
        perturbation=noise_rng.normal(size=cfg.steps)
        for history in (0,1):
            rng=np.random.default_rng(np.random.SeedSequence([seed,split,pair,91]))
            drives=[make_drive(sample_cue(history,rng,cfg,evaluation=split==2),cfg) for _ in range(history_pulses)]
            for d in drives:digest.update(d.tobytes())
            material=write_history(grid,initial,drives,cfg,rate=write_rate)
            trace=probe_waveform(grid,material,probe,cfg)
            raw.append(trace+noise*np.std(trace)*perturbation)
            erased.append(baseline+noise*np.std(baseline)*perturbation)
            labels.append(history)
            if pair==0:examples.append({'history':history,'material':material.tolist(),'raw_trace':trace.tolist()})
    return {'raw':np.asarray(raw),'erased':np.asarray(erased),'labels':np.asarray(labels),
        'initial_material':initial.tolist(),'examples':examples,'probe_drive':probe,
        'history_digest':digest.hexdigest(),
        'memory':{'box1_persistent_scalars':len(initial),'box1_persistent_bytes':initial.nbytes,
            'box1_fast_state_at_probe':'zero, independently for every history',
            'transport_samples':cfg.steps,'transport_bytes':cfg.steps*8,
            'conditioner':'offline full-window affine matching; buffer counted separately'}}
