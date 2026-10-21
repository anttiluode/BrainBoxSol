"""Positive, resource-normalized local mobility and an equal-state trace control.

An engineered phenomenological rule; not variable-capacitor electrodynamics.
Every rollout starts with zero fast voltage. No labels enter these functions.
"""
from __future__ import annotations
import numpy as np


def validate_material(material):
    m=np.asarray(material,dtype=float)
    if m.ndim!=2 or not m.size or not np.isfinite(m).all() or np.any(m<=0):
        raise ValueError("positive finite batched material required")
    return m


def validate_operator(A,dt,nodes):
    A=np.asarray(A,dtype=float)
    if (A.shape!=(nodes,nodes) or not np.isfinite(A).all()
            or not np.allclose(A,A.T,atol=1e-14,rtol=0)
            or not np.isfinite(dt) or dt<=0 or np.any(np.diag(A)<=0)
            or np.any((A-np.diag(np.diag(A)))>1e-14) or np.any(A.sum(axis=1)<-1e-14)
            or dt*2*np.max(np.diag(A))>.8+1e-14):
        raise ValueError("invalid RC operator or unstable time step")
    return A


def steps_for(duration,dt):
    if not np.isfinite([duration,dt]).all() or duration<0 or dt<=0:
        raise ValueError("invalid duration/time step")
    steps=int(round(duration/dt))
    if not np.isclose(steps*dt,duration,rtol=0,atol=1e-10):
        raise ValueError("duration must be an integer number of steps")
    return steps


def mobility(m,local):
    r=1/(1+m)
    return r if local else r.mean(axis=1,keepdims=True)


def voltage_step(A,v,m,drive,dt):
    m=validate_material(m);A=validate_operator(A,dt,m.shape[1])
    v=np.asarray(v,dtype=float);drive=np.asarray(drive,dtype=float)
    if v.shape!=m.shape or drive.shape!=m.shape or not np.isfinite(v).all() or not np.isfinite(drive).all():
        raise ValueError("voltage/current shape or finiteness mismatch")
    return v+dt*mobility(m,True)*(drive-v@A.T)


def update_material(m,target,r,amount):
    # A convex positive update followed by an explicit global resource controller.
    budget=m.mean(axis=1,keepdims=True)
    result=m+amount*r*(target-m)
    return result*(budget/result.mean(axis=1,keepdims=True))


def evolve(A,material,drive,*,dt,rate,local,floor=.1):
    m=validate_material(material).copy();A=validate_operator(A,dt,m.shape[1])
    drive=np.asarray(drive,dtype=float)
    if (drive.ndim!=3 or drive.shape[0]!=len(m) or drive.shape[2]!=m.shape[1]
            or not np.isfinite(drive).all() or not np.isfinite([rate,floor]).all()
            or rate<0 or dt*rate>1 or floor<=0):
        raise ValueError("invalid drive or plasticity parameters")
    v=np.zeros_like(m);voltage_steps=material_steps=0
    budget=m.mean(axis=1,keepdims=True)
    for current in drive.transpose(1,0,2):
        r=mobility(m,local)
        v+=dt*r*(current-v@A.T)
        voltage_steps+=len(m)
        if rate:
            energy=v*v;average=energy.mean(axis=1,keepdims=True)
            normalized=np.divide(energy,average,out=np.ones_like(energy),where=average>1e-20)
            target=budget*(floor+normalized)/(1+floor)
            m=update_material(m,target,r,dt*rate)
            material_steps+=len(m)
    if not np.isfinite(v).all() or not np.isfinite(m).all():
        raise ValueError("nonfinite evolution")
    return m,v,{"voltage_vector_steps":voltage_steps,"material_vector_steps":material_steps}


def wait_material(material,duration,*,dt,rate,local):
    m=validate_material(material).copy();steps=steps_for(duration,dt)
    if not np.isfinite(rate) or rate<0 or dt*rate>1:
        raise ValueError("invalid waiting rate")
    target=np.broadcast_to(m.mean(axis=1,keepdims=True),m.shape)
    for _ in range(steps):
        m=update_material(m,target,mobility(m,local),dt*rate)
    return m,steps*len(m)


def probe(A,material,ports,cfg):
    m=validate_material(material);dt=cfg["dt"];A=validate_operator(A,dt,m.shape[1])
    steps=steps_for(cfg["probe_duration"],dt);pulse=steps_for(cfg["probe_pulse"],dt)
    samples=[steps_for(t,dt) for t in cfg["sample_times"]]
    ports=np.asarray(ports,dtype=int)
    if (not samples or min(samples)<1 or max(samples)>steps or len(set(samples))!=len(samples)
            or ports.ndim!=1 or np.any(ports<0) or np.any(ports>=m.shape[1])
            or not np.isfinite(cfg["probe_amplitude"]) or cfg["probe_amplitude"]<=0):
        raise ValueError("invalid probe boundary")
    v=np.zeros_like(m);r=mobility(m,True);values={}
    for t in range(steps):
        dv=-v@A.T
        if t<pulse:dv[:,m.shape[1]//2]+=cfg["probe_amplitude"]
        v+=dt*r*dv
        if t+1 in samples:values[t+1]=v[:,ports].copy()
    return np.concatenate([values[t] for t in samples],axis=1),steps*len(m)
