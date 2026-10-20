"""Positive spatial conductance medium. Activity is reset; material persists."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    size: int
    edges: np.ndarray

    @property
    def nodes(self):
        return self.size * self.size


def make_grid(size=9):
    if not isinstance(size, int) or isinstance(size, bool) or size < 2:
        raise ValueError('grid size must be an integer >=2')
    edges = []
    for r in range(size):
        for c in range(size):
            for dr, dc in ((0,1), (1,0), (1,1), (1,-1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < size and 0 <= cc < size:
                    edges.append((r*size+c, rr*size+cc))
    return Grid(size, np.array(edges, dtype=np.int64))


def project_budget(values, budget, lower, upper):
    x = np.asarray(values, dtype=np.float64)
    if (x.ndim != 1 or not x.size or not np.isfinite(x).all()
            or not np.isfinite([budget, lower, upper]).all()
            or lower < 0 or upper <= lower
            or not x.size*lower <= budget <= x.size*upper):
        raise ValueError('infeasible or nonfinite material projection')
    lo, hi = float(x.min()-upper), float(x.max()-lower)
    for _ in range(100):
        mid = (lo+hi)/2
        y = np.clip(x-mid, lower, upper)
        if abs(y.sum()-budget) <= 1e-13*max(budget, 1e-3):
            break
        if y.sum() > budget: lo = mid
        else: hi = mid
    if abs(y.sum()-budget) > 1e-10*max(budget, 1e-3):
        raise ValueError('material projection failed to converge')
    return y


def initial_material(grid, seed, *, mean=.08, lower=.005, upper=.4):
    x = mean*np.random.default_rng(seed).uniform(.9,1.1,len(grid.edges))
    return project_budget(x, len(x)*mean, lower, upper)


def operator(grid, g, *, leak):
    g = np.asarray(g, dtype=np.float64)
    if (g.shape != (len(grid.edges),) or not np.isfinite(g).all()
            or np.any(g <= 0) or not np.isfinite(leak) or leak < 0):
        raise ValueError('positive finite conductances and nonnegative leak required')
    a, b = grid.edges.T
    A = leak*np.eye(grid.nodes)
    np.add.at(A, (a,a), g); np.add.at(A, (b,b), g)
    A[a,b] -= g; A[b,a] -= g
    return A


def transition(grid, g, *, dt, leak):
    A = operator(grid, g, leak=leak)
    degree = np.diag(A)-leak
    if not np.isfinite(dt) or dt <= 0 or dt*(2*degree.max()+leak) > .8+1e-14:
        raise ValueError('time step violates conservative stability bound')
    return np.eye(grid.nodes)-dt*A


def rollout(grid, g, drive, *, dt, leak):
    M = transition(grid, g, dt=dt, leak=leak)
    drive = np.asarray(drive, dtype=np.float64)
    if drive.ndim != 2 or drive.shape[1] != grid.nodes or not np.isfinite(drive).all():
        raise ValueError('drive must be finite with shape (steps,nodes)')
    states = np.zeros((len(drive)+1, grid.nodes))
    for t in range(len(drive)):
        states[t+1] = M@states[t] + dt*drive[t]
    if not np.isfinite(states).all():
        raise ValueError('nonfinite activity')
    return states


def loss_gradient(grid, g, drive, cfg, target_sign):
    """Exact reverse-mode derivative of the finite discrete dynamics."""
    from adaptive_medium_task import read_motor, motor_masks
    if target_sign not in (-1,1): raise ValueError('target sign must be -1 or +1')
    states=rollout(grid,g,drive,dt=cfg.dt,leak=cfg.leak)
    episode=read_motor(states,cfg)
    e0,e1=episode.energies
    den=e0+e1+cfg.ratio_epsilon
    r=(e0-e1)/den
    residual=r-cfg.target_magnitude*target_sign
    derivative=residual*np.array([2*e1+cfg.ratio_epsilon,-2*e0-cfg.ratio_epsilon])/den**2
    q=np.zeros_like(states)
    q[cfg.read_start:]=cfg.dt*(derivative@motor_masks(cfg))
    adjoint=np.empty_like(states); adjoint[-1]=q[-1]
    M=transition(grid,g,dt=cfg.dt,leak=cfg.leak)
    for t in range(cfg.steps-1,-1,-1):
        adjoint[t]=q[t]+M.T@adjoint[t+1]
    a,b=grid.edges.T
    forward_drop=states[:-1,a]-states[:-1,b]
    return_drop=adjoint[1:,a]-adjoint[1:,b]
    gradient=-cfg.dt*np.einsum('te,te->e',forward_drop,return_drop)
    if not np.isfinite(gradient).all(): raise ValueError('nonfinite material gradient')
    return float(.5*residual**2),episode,gradient


def motor_response_map(grid,g,cfg):
    """Exact two-motor boundary map for the specified pulse and time window."""
    from adaptive_medium_task import motor_masks
    M=transition(grid,g,dt=cfg.dt,leak=cfg.leak)
    q=cfg.dt*motor_masks(cfg)
    lam=q.copy(); result=np.zeros_like(q)
    for t in range(cfg.steps-1,-1,-1):
        if t<cfg.pulse: result+=cfg.dt*lam
        lam=lam@M+(q if t>=cfg.read_start else 0)
    return result
