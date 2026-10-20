"""One-bit reward writes via paired parameter perturbations, without an adjoint.

The environment holds labels. RewardWriter sees only proposed material and
binary outcome arrays. This is a black-box search method, not a biological law.
"""
from __future__ import annotations
import hashlib
import time
import numpy as np
from adaptive_medium import initial_material,project_budget,transition
from adaptive_medium_task import make_drive,motor_masks,episode_from_energies,sample_cue
from adaptive_medium_gate import MaterialRun,validate_material,check_training


class RewardWriter:
    def __init__(self,material,cfg,*,seed,pairs=8,sigma=.04,rate=.01):
        from adaptive_medium import make_grid
        self.material=validate_material(material,make_grid(cfg.size),cfg).copy()
        if not isinstance(pairs,int) or pairs<2 or pairs%2 or not np.isfinite([sigma,rate]).all() or sigma<=0 or rate<=0:
            raise ValueError('positive finite rates and an even pair count >=2 required')
        self.cfg=cfg;self.pairs=pairs;self.sigma=sigma;self.rate=rate
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,70]))
        self.random_rng=np.random.default_rng(np.random.SeedSequence([seed,74]))
        self.eligibility=None

    def propose(self):
        if self.eligibility is not None:raise ValueError('pending proposals need feedback first')
        proposals=[]
        for _ in range(self.pairs):
            noise=self.rng.choice([-1.,1.],len(self.material))
            proposals.append([project_budget(self.material+sign*self.sigma*noise,
                len(self.material)*self.cfg.mean,self.cfg.lower,self.cfg.upper) for sign in (1,-1)])
        p=np.asarray(proposals)
        self.eligibility=(p[:,0]-p[:,1])/(2*self.sigma)
        return p

    def observe(self,reward_bits,*,random_direction=False):
        bits=np.asarray(reward_bits,dtype=float)
        if (self.eligibility is None or bits.shape!=(self.pairs,2)
                or not np.isin(bits,[0,1]).all()):
            raise ValueError('one 0/1 outcome per pending proposal required')
        delta=self.rate*np.mean((bits[:,0]-bits[:,1])[:,None]*self.eligibility,axis=0)/(2*self.sigma)
        if random_direction:
            direction=self.random_rng.normal(size=len(delta)); direction-=direction.mean()
            delta=np.linalg.norm(delta)*direction/np.linalg.norm(direction)
        self.material=project_budget(self.material+delta,len(self.material)*self.cfg.mean,
                                     self.cfg.lower,self.cfg.upper)
        self.eligibility=None


def reward_trial(grid,g,source,cfg):
    """Stream one physical forward trial; no trajectory or backward solver."""
    drive=make_drive(source,cfg)
    M=transition(grid,g,dt=cfg.dt,leak=cfg.leak)
    u=np.zeros(grid.nodes);energy=np.zeros(2);masks=motor_masks(cfg)
    for t in range(cfg.steps):
        u=M@u+cfg.dt*drive[t]
        if t+1>=cfg.read_start:energy+=cfg.dt*(masks@u)
    return episode_from_energies(energy,cfg)


def train_reward_material(grid,seed,mapping,cfg,*,updates=512,pairs=8,sigma=.04,rate=.01,
                          temperature=5.,mode='reward'):
    """Environment wrapper. Only this function may use mapping to grade actions."""
    check_training(mapping,updates,rate)
    if mode not in ('reward','shuffled_reward','random_updates') or not np.isfinite(temperature) or temperature<=0:
        raise ValueError('invalid reward mode or policy temperature')
    start=time.perf_counter();g=initial_material(grid,seed,mean=cfg.mean,lower=cfg.lower,upper=cfg.upper)
    writer=RewardWriter(g,cfg,seed=seed,pairs=pairs,sigma=sigma,rate=rate)
    sources=np.random.default_rng(np.random.SeedSequence([seed,72]))
    choices=np.random.default_rng(np.random.SeedSequence([seed,71]))
    shuffle=np.random.default_rng(np.random.SeedSequence([seed,73]))
    digest=hashlib.sha256(); curve=[];invalid=0;trial_count=0;true_total=0
    for step in range(updates):
        proposals=writer.propose();bits=np.zeros((pairs,2),dtype=int)
        for j in range(pairs):
            cue=j%2;source=sample_cue(cue,sources,cfg,evaluation=False)
            coin=choices.random()  # common random number for the two antithetic probes
            for k in (0,1):
                digest.update(source.tobytes());ep=reward_trial(grid,proposals[j,k],source,cfg)
                probability=1/(1+np.exp(-np.clip(temperature*ep.score,-40,40)))
                action=0 if coin<probability else 1
                bits[j,k]=int(ep.valid and action==mapping[cue])
                invalid+=int(not ep.valid);trial_count+=1
        true_total+=int(bits.sum());feedback=bits.copy()
        if mode=='shuffled_reward':feedback=shuffle.permutation(feedback.ravel()).reshape(pairs,2)
        writer.observe(feedback,random_direction=mode=='random_updates')
        if step%10==0 or step==updates-1:
            curve.append({'update':step+1,'true_reward_fraction':true_total/trial_count,
                          'batch_reward_fraction':float(bits.mean())})
    costs={'seconds':time.perf_counter()-start,'forward_rollouts':trial_count,'adjoint_rollouts':0,
        'solver_steps':trial_count*cfg.steps,'feedback_bits':trial_count,'invalid_training_trials':invalid,
        'input_digest':digest.hexdigest(),'persistent_scalars':len(g),'persistent_bytes':g.nbytes,
        'peak_trajectory_bytes':0,'drive_bytes':cfg.steps*grid.nodes*8,
        'memory_note':'Array payload estimates; excludes Python overhead and transient NumPy projection allocations.','proposal_and_eligibility_bytes':3*pairs*len(g)*8,
        'state_and_operator_bytes':(grid.nodes**2+grid.nodes+2)*8,
        'source_schedule':'balanced pairs; identical source and policy coin within each pair',
        'learning_signal':'binary outcomes only; common-coin paired parameter perturbations'}
    return MaterialRun(g,writer.material.copy(),curve,costs)
