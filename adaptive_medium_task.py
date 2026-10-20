"""Fixed sensory/motor task; no learned readout and no target in the actor."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    size: int = 9
    dt: float = .1
    leak: float = .02
    mean: float = .08
    lower: float = .005
    upper: float = .4
    steps: int = 160
    pulse: int = 24
    read_start: int = 113
    width: float = .7
    train_jitter: float = .25
    eval_jitter: float = .35
    train_noise: float = .05
    eval_noise: float = .10
    amplitude_min: float = .8
    amplitude_max: float = 1.2
    ratio_epsilon: float = 1e-9
    energy_floor: float = 1e-8
    target_magnitude: float = .8

    def __post_init__(self):
        from dataclasses import asdict
        if (not all(np.isfinite(v) for v in asdict(self).values()) or self.size != 9
                or self.dt <= 0 or self.leak < 0 or not 0 < self.lower <= self.mean <= self.upper
                or self.upper <= self.lower or self.width <= 0
                or not all(isinstance(x,int) for x in (self.steps,self.pulse,self.read_start))
                or not 1 <= self.pulse <= self.steps or not 1 <= self.read_start <= self.steps
                or min(self.train_jitter,self.eval_jitter,self.train_noise,self.eval_noise) < 0
                or not 0 < self.amplitude_min <= self.amplitude_max
                or self.energy_floor <= 0 or self.ratio_epsilon <= 0
                or not 0 < self.target_magnitude < 1):
            raise ValueError('invalid sensory task configuration')


@dataclass
class Episode:
    score: float
    energies: np.ndarray
    action: int | None
    valid: bool
    failure: str | None


def sample_cue(cue, rng, cfg, *, evaluation):
    if cue not in (0,1): raise ValueError('cue must be 0 or 1')
    row = (2,6)[cue]
    jitter = cfg.eval_jitter if evaluation else cfg.train_jitter
    noise = cfg.eval_noise if evaluation else cfg.train_noise
    dy, dx = rng.uniform(-jitter,jitter,2)
    yy,xx=np.mgrid[-1:2,-1:2]
    patch=np.exp(-((yy-dy)**2+(xx-dx)**2)/(2*cfg.width**2))
    patch=np.maximum(0,patch+rng.normal(0,noise*patch.max(),patch.shape))
    if patch.sum() == 0: raise ValueError('empty sensory source')
    patch=patch/patch.sum()*rng.uniform(cfg.amplitude_min,cfg.amplitude_max)
    source=np.zeros((cfg.size,cfg.size)); source[row-1:row+2,:3]=patch
    return source.ravel()


def make_drive(source,cfg):
    source=np.asarray(source,dtype=float)
    if source.shape!=(cfg.size**2,) or not np.isfinite(source).all() or np.any(source<0):
        raise ValueError('invalid sensory source')
    drive=np.zeros((cfg.steps,cfg.size**2)); drive[:cfg.pulse]=source
    return drive


def motor_masks(cfg):
    masks=np.zeros((2,cfg.size**2))
    masks[0,np.arange(1,4)*cfg.size+7]=1
    masks[1,np.arange(5,8)*cfg.size+7]=1
    return masks


def episode_from_energies(energies,cfg):
    e=np.asarray(energies,dtype=float).copy()
    if e.shape!=(2,): raise ValueError('motor energies must have shape (2,)')
    if not np.isfinite(e).all() or np.any(e<0) or e.sum()<=cfg.energy_floor:
        return Episode(0.,e,None,False,'invalid or low motor energy')
    r=float((e[0]-e[1])/(e.sum()+cfg.ratio_epsilon))
    return Episode(r,e,0 if r>0 else (1 if r<0 else None),True,None if r!=0 else 'tie')


def read_motor(states,cfg):
    states=np.asarray(states,dtype=float)
    if states.shape!=(cfg.steps+1,cfg.size**2) or not np.isfinite(states).all():
        raise ValueError('invalid recorded activity')
    energy=cfg.dt*(states[cfg.read_start:].sum(axis=0)@motor_masks(cfg).T)
    return episode_from_energies(energy,cfg)
