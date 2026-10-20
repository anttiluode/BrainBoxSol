"""Phase–bridge integration: does PAC-like timing protect learned material readouts?

A deliberately small six-block network; not a replication of BrainBoxClaude's
192-node G2/G3. Reuses BrainBoxSol's supervised protected writer.
Run: OPENBLAS_NUM_THREADS=1 python phase_bridge_gate.py
"""
import json
from pathlib import Path
from itertools import combinations
import numpy as np
from brainboxsol import write_protected


class PhaseMedium:
    def __init__(self, blocks=6, inside=.8, bridge=.035, leak=.001, seed=0):
        rng = np.random.default_rng(900+seed)
        self.blocks = blocks
        edges, owner, interface = [], [], []
        for k in range(blocks):
            b = 4*k
            for a, c in ((0,1), (0,2), (1,3), (2,3), (1,2)):
                edges.append((b+a,b+c)); owner.append(k); interface.append(-1)
            if k < blocks-1:
                for a, c in ((3,0), (1,0), (3,2)):
                    edges.append((b+a,b+4+c)); owner.append(-1); interface.append(k)
        self.owner = np.array(owner)
        self.interface = np.array(interface)
        self.incidence = np.zeros((4*blocks, len(edges)))
        for z, (a,b) in enumerate(edges):
            self.incidence[a,z] = 1
            self.incidence[b,z] = -1
        self.initial = np.where(self.owner == -1, bridge, inside).astype(float)
        self.initial[self.interface >= 0] *= np.tile([1.2,1.,.8],blocks-1)
        self.initial[self.owner >= 0] *= np.exp(
            .1*rng.standard_normal(np.count_nonzero(self.owner >= 0)))
        self.initial[self.interface >= 0] *= np.exp(
            .2*rng.standard_normal(np.count_nonzero(self.interface >= 0)))
        self.leak = leak
        self.ports = np.arange(blocks)*4+3
        self.soma = 0

    def matrix(self, g):
        B = self.incidence
        return (B*g)@B.T + self.leak*np.eye(B.shape[0])

    def read(self, g, k):
        A = self.matrix(g)
        b = np.zeros(A.shape[0]); b[self.ports[k]] = 1
        c = np.zeros_like(b); c[self.soma] = 1
        f = np.linalg.solve(A,b)
        a = np.linalg.solve(A,c)
        grad = -(self.incidence.T@f)*(self.incidence.T@a)
        return float(f[self.soma]), grad

    def values(self, g):
        return np.array([self.read(g,k)[0] for k in range(self.blocks)])

    def transfer(self, g, omega):
        drive = np.zeros(self.incidence.shape[0]); drive[self.soma] = 1
        return np.linalg.solve(self.matrix(g)+1j*omega*np.eye(len(drive)), drive)

    def phases(self, g, omega):
        return np.angle(self.transfer(g,omega)[self.ports])


def teacher(box, signs):
    out = box.initial.copy()
    for k, sgn in enumerate(signs):
        out[box.owner == k] *= 1+.4*sgn
    return out


def fit(box, signs, order=None, steps=24):
    if order is None:
        order = list(range(box.blocks))
    target = box.values(teacher(box,signs))
    baseline = box.values(box.initial)
    state = box.initial.copy()
    old = []
    for k in order:
        for _ in range(steps):
            write_protected(box,state,k,target,baseline,old)
        old.append(k)
    return state, target


def drift(box,g,sd,rng):
    out = g.copy()
    chosen = box.interface >= 0
    out[chosen] *= np.exp(sd*rng.standard_normal(np.count_nonzero(chosen)))
    return out


def homeostat(box,g,baseline,method):
    out = g.copy()
    if method == 'interface':
        for k in range(box.blocks-1):
            chosen = box.interface == k
            out[chosen] *= baseline[chosen].sum()/out[chosen].sum()
    elif method == 'global':
        chosen = box.interface >= 0
        out[chosen] *= baseline[chosen].sum()/out[chosen].sum()
    elif method == 'pernode':
        E = box.incidence
        target = (abs(E)*baseline).sum(axis=1)
        for _ in range(100):
            d = (abs(E)*out).sum(axis=1)
            factor = np.sqrt(target/np.maximum(d,1e-12))
            scale = np.prod(np.where(E.T != 0,factor,1),axis=1)
            out *= scale
    elif method != 'none':
        raise ValueError(method)
    return out


def templates(box,g,omega,ts,mode,gate_g=None):
    if gate_g is None:
        gate_g = g
    if mode == 'linear':
        release = np.zeros(box.blocks)
    elif mode == 'timed':
        release = np.arange(box.blocks)/box.blocks*(2*np.pi/omega)
    elif mode in ('phase','phase_frozen'):
        release = np.mod(-box.phases(gate_g,omega),2*np.pi)/omega
    else:
        raise ValueError(mode)
    eigen,U = np.linalg.eigh(box.matrix(g))
    weights = U[box.soma,:][None,:]*U[box.ports,:]
    t = np.maximum(ts[None,:]-release[:,None],0.)
    response = (weights[:,:,None]*np.exp(-eigen[None,:,None]*t[:,None,:])).sum(1)
    response[ts[None,:] < release[:,None]] = 0
    return response, release


def decode(templates,y):
    # Known K=2. Brute force all two-source sets; clipped least-squares amplitudes.
    best = None
    for chosen in combinations(range(len(templates)),2):
        A = templates[list(chosen)].T
        coeff = np.maximum(np.linalg.lstsq(A,y,rcond=None)[0],0)
        error = np.linalg.norm(A@coeff-y)
        if best is None or error < best[0]:
            best = (error,chosen,coeff)
    return best


def trial(box,g0,gdrift,omega,ts,mode,rng,rep=50,noise=.012):
    if mode in ('phase','phase_frozen'):
        dictionary,_ = templates(box,g0,omega,ts,mode,gate_g=g0)
        actual,_ = templates(box,gdrift,omega,ts,mode,
                             gate_g=gdrift if mode == 'phase' else g0)
    else:
        dictionary,_ = templates(box,g0,omega,ts,mode)
        actual,_ = templates(box,gdrift,omega,ts,mode)
    scale = np.sqrt((dictionary**2).mean())
    correct = 0
    errors = []
    for _ in range(rep):
        pair = rng.choice(box.blocks,2,replace=False)
        amplitude = rng.uniform(.6,1.4,2)
        y = amplitude@actual[pair]+noise*scale*rng.standard_normal(len(ts))
        _, found, coeff = decode(dictionary,y)
        correct += set(found) == set(pair)
        truth = np.zeros(box.blocks); truth[pair] = amplitude
        estimate = np.zeros(box.blocks); estimate[list(found)] = coeff
        errors.append(np.linalg.norm(estimate-truth)/np.linalg.norm(truth))
    return correct/rep, float(np.mean(errors))


def run_sweep(seeds=range(5),drift_strengths=(.6,1.0),trials=50):
    """Post-hoc, fixed known-port, supervised material + PAC-like timing study."""
    conditions = ('none','global','interface','pernode')
    modes = ('linear','phase','phase_frozen','timed')
    records = []
    for seed in seeds:
        rng = np.random.default_rng(514+seed)
        box = PhaseMedium(seed=seed)
        omega = .012
        ts = np.linspace(0,2*np.pi/omega+500,500)
        signs = np.where(rng.random(box.blocks)<.5,1,-1)
        order = list(rng.permutation(box.blocks))
        trained, targets = fit(box,signs,order=order)
        baseline = box.values(box.initial)
        training_error = float(np.mean(np.abs(box.values(trained)-targets)/baseline))
        for strength in drift_strengths:
            damaged = drift(box,trained,strength,rng)
            for method in conditions:
                recovered = homeostat(box,damaged,trained,method)
                response_error = float(np.mean(np.abs(box.values(recovered)-targets)/baseline))
                phase_shift = np.degrees(np.angle(
                    np.exp(1j*(box.phases(recovered,omega)-box.phases(trained,omega)))))
                interface_diff = max(
                    abs(recovered[box.interface==k].sum()-trained[box.interface==k].sum())
                    for k in range(box.blocks-1))
                for mode in modes:
                    accuracy,relative_error = trial(
                        box,trained,recovered,omega,ts,mode,
                        np.random.default_rng(2040+seed),trials)
                    records.append(dict(
                        seed=int(seed),drift=float(strength),homeostat=method,
                        read=mode,source_pair_accuracy=accuracy,
                        amplitude_relative_error=relative_error,
                        stored_response_error=response_error,
                        mean_abs_phase_shift_deg=float(np.abs(phase_shift).mean()),
                        max_interface_total_error=float(interface_diff),
                        pre_drift_train_error=training_error,
                        teacher_signs=signs.tolist(),
                        order=[int(k) for k in order]))
    summary = {}
    for strength in drift_strengths:
        for method in conditions:
            for mode in modes:
                group = [r for r in records if r['drift']==strength
                         and r['homeostat']==method and r['read']==mode]
                summary[f'{strength}:{method}:{mode}'] = {
                    key:float(np.mean([r[key] for r in group]))
                    for key in ('source_pair_accuracy','amplitude_relative_error',
                                'stored_response_error','mean_abs_phase_shift_deg',
                                'max_interface_total_error')}
    return {
        'design':{
            'seeds':list(seeds),'drift_strengths':list(drift_strengths),
            'trials_per_condition_per_seed':trials,
            'blocks':6,'cells_per_block':4,'bridge_edges_per_interface':3,
            'omega':.012,'noise':.012,'two_simultaneous_active_blocks':True,
            'trained_writer':'supervised Sol OGD/adjoint protected writer',
            'readers':'stale pre-drift templates; K=2 known, exhaustive 2-support least squares',
            'modes':{
                'linear':'simultaneous pulses at time zero',
                'phase':'PAC-like release using CURRENT local phase',
                'phase_frozen':'externally fixed OLD phase schedule',
                'timed':'externally assigned evenly spaced block slots'},
            'posthoc':True},
        'summary':summary,'cases':records,
        'limits':[
            'Imposed phase-dependent release, not trained or biological PAC.',
            'Stale templates know candidate ports and the correct active-set size K=2.',
            'Interface homeostat knows pre-drift total of each bridge interface.',
            'Fixed-slot control uses externally assigned positional codes.',
            'No new memory-capacity, biological lipid or faster-AI claim.',
            'The protected writer has supervised targets, old-query replay and SVD cost.',
            'The chosen drift is multiplicative bridge-only noise.',
            'Post-hoc five-seed toy rather than an independent held-out benchmark.'
        ]
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/phase_bridge.json'))
    args = p.parse_args()
    result = run_sweep()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('drift homeostat reader: correct_pair / relative_amp_error / stored_response_error / phase_deg')
    for label,metrics in result['summary'].items():
        print(f"{label:35s} {metrics['source_pair_accuracy']:.3f} / "
              f"{metrics['amplitude_relative_error']:.3f} / "
              f"{metrics['stored_response_error']:.4f} / "
              f"{metrics['mean_abs_phase_shift_deg']:.1f}")
    print('Raw receipt:',args.out)


if __name__ == '__main__':
    main()
