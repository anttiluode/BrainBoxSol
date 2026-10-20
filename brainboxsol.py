"""BrainBoxSol: spectral scaffold and local adjoint material writes.

Toy passive resistor/diffusion medium; not a neuron or lipid-zone model.
Run: python brainboxsol.py. Dependency: numpy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Medium:
    blocks: int = 4
    intra: float = 0.9
    bridge: float = 0.055
    leak: float = 0.035

    def __post_init__(self):
        edges, owner = [], []
        for k in range(self.blocks):
            base = 4 * k
            # Two internal paths connect inlet to the block's tip.
            for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
                edges.append((base + a, base + b))
                owner.append(k)
            if k < self.blocks - 1:
                edges.append((base + 3, base + 4))
                owner.append(-1)  # slow, protected address bridges
        n = 4 * self.blocks
        incidence = np.zeros((n, len(edges)))
        for e, (a, b) in enumerate(edges):
            incidence[a, e] = 1
            incidence[b, e] = -1
        self.incidence = incidence
        self.owner = np.asarray(owner)
        self.initial = np.where(self.owner == -1, self.bridge, self.intra).astype(float)
        self.ports = np.arange(self.blocks) * 4 + 3
        self.soma = 0

    def matrix(self, conductances):
        B = self.incidence
        return (B * conductances) @ B.T + self.leak * np.eye(B.shape[0])

    def read(self, conductances, block):
        """Steady soma voltage from a unit current injected at one block tip."""
        A = self.matrix(conductances)
        source = np.zeros(A.shape[0]); source[self.ports[block]] = 1
        soma = np.zeros(A.shape[0]); soma[self.soma] = 1
        forward = np.linalg.solve(A, source)
        adjoint = np.linalg.solve(A, soma)
        # Physical edge sensitivity: -(voltage drop)*(adjoint drop).
        delta_f = self.incidence.T @ forward
        delta_a = self.incidence.T @ adjoint
        derivative = -delta_f * delta_a
        return float(forward[self.soma]), derivative

    def values(self, conductances):
        return np.array([self.read(conductances, k)[0] for k in range(self.blocks)])

    def impulse(self, conductances, block, times):
        """Impulse response from a freshly cleared fast voltage state."""
        eig, modes = np.linalg.eigh(self.matrix(conductances))
        weights = modes[self.soma, :] * modes[self.ports[block], :]
        return np.exp(-np.outer(times, eig)) @ weights

    def spectrum(self, conductances):
        return np.linalg.eigvalsh(self.matrix(conductances))


def teacher_medium(box, signs=(1, -1, 1, -1)):
    """Make realizable targets; teacher material remains hidden from writer."""
    teacher = box.initial.copy()
    for k, direction in enumerate(signs):
        teacher[box.owner == k] *= 1 + 0.4 * direction
    return teacher


def write_one(box, state, block, target, baseline, mode='local', rate=.15):
    """One *supervised* adjoint step; no claim about unsupervised pings.

    -dLoss/dg uses a scalar target error and local forward * adjoint drops.
    Slope normalization equalizes near/far gradient scale. No bridge writes.
    """
    response, sensitivity = box.read(state, block)
    residual = (response - target[block]) / baseline[block]
    gradient = sensitivity / baseline[block]
    writable = (box.owner == block) if mode == 'local' else (box.owner >= 0)
    norm2 = np.dot(gradient[writable], gradient[writable])
    state[writable] -= rate * residual * gradient[writable] / max(norm2, 1e-12)
    state[writable] = np.clip(state[writable], .1, 2.)


def sequential_training(box, target, baseline, mode, steps=80):
    state = box.initial.copy()
    histories = []
    for k in range(box.blocks):
        for _ in range(steps):
            if mode != 'frozen':
                write_one(box, state, k, target, baseline, mode)
        histories.append(((box.values(state) - target) / baseline).tolist())
    return state, np.array(histories)


def run_gate(steps=80, drift=.2):
    box = Medium()
    baseline = box.values(box.initial)
    target = box.values(teacher_medium(box))
    state_l, hist_l = sequential_training(box, target, baseline, 'local', steps)
    state_g, hist_g = sequential_training(box, target, baseline, 'global', steps)
    _, hist_f = sequential_training(box, target, baseline, 'frozen', steps)

    # Analytic adjoint gradients versus fresh finite differences.
    numerical, analytic = [], []
    for k in range(box.blocks):
        y, sensitivity = box.read(box.initial, k)
        for edge in np.flatnonzero(box.owner >= 0):
            perturbed = box.initial.copy()
            perturbed[edge] += 1e-5
            numerical.append((box.read(perturbed, k)[0] - y) / 1e-5)
            analytic.append(sensitivity[edge])
    finite_difference_max = float(np.max(np.abs(np.array(numerical) - np.array(analytic))))

    # Genuine bridge drift, not a boundary-preserving gauge transform.
    moved = state_l.copy()
    moved[box.owner == -1] *= np.array([1 - drift, 1 + drift, 1 - drift])
    pre_drift = (box.values(state_l) - target) / baseline
    post_drift = (box.values(moved) - target) / baseline
    # A supervised repair attempt has explicit access to all old targets.
    recovered = moved.copy()
    for step in range(160):
        write_one(box, recovered, step % box.blocks, target, baseline, 'local')
    post_replay = (box.values(recovered) - target) / baseline

    eig = box.spectrum(box.initial)
    result = {
        'protocol': {'blocks': box.blocks, 'physical_nodes': 4 * box.blocks,
                     'writable_edges': int(np.sum(box.owner >= 0)),
                     'bridge_edges': int(np.sum(box.owner == -1)),
                     'steps_per_task': steps, 'drift_fraction': drift,
                     'teacher_pattern': [1, -1, 1, -1]},
        'initial_eigenvalues': [float(x) for x in eig],
        'macro_spectral_gap_ratio': float(eig[4] / eig[3]),
        'baseline': baseline.tolist(), 'targets': target.tolist(),
        'finite_difference_max_abs_error': finite_difference_max,
        'sequential_relative_error': {
            'local': hist_l.tolist(), 'global': hist_g.tolist(), 'frozen': hist_f.tolist()},
        'final_mean_abs_relative_error': {
            'local': float(np.mean(abs(hist_l[-1]))),
            'global': float(np.mean(abs(hist_g[-1]))),
            'frozen': float(np.mean(abs(hist_f[-1])))},
        'pre_drift_error': pre_drift.tolist(),
        'post_local_bridge_drift_error': post_drift.tolist(),
        'supervised_replay_error': post_replay.tolist(),
        'bridge_unchanged_during_training': bool(
            np.array_equal(state_l[box.owner == -1], box.initial[box.owner == -1])
            and np.array_equal(state_g[box.owner == -1], box.initial[box.owner == -1])),
        'direct_table_relative_error': 0.,
        'warnings': [
            'Teacher supplies supervised scalar targets; not an unsupervised ping.',
            'Static known input ports serve as addresses; no learned PAC addressing.',
            'Macro modes are decays, not oscillations or phase-amplitude coupling.',
            'Direct indexed table stores four targets exactly and is smaller.',
            'One toy teacher, one training order, no broad performance claim.',
            'Replay with old targets is supervised repair, not autonomous homeostasis.'
        ]
    }
    return result


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('results/gate0.json'))
    p.add_argument('--steps', type=int, default=80)
    args = p.parse_args()
    receipt = run_gate(steps=args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'spectral_gap_ratio': receipt['macro_spectral_gap_ratio'],
        'finite_difference_error': receipt['finite_difference_max_abs_error'],
        'mean_abs_error': receipt['final_mean_abs_relative_error'],
        'post_drift': [round(x, 4) for x in receipt['post_local_bridge_drift_error']],
        'after_supervised_replay': [round(x, 4) for x in receipt['supervised_replay_error']],
        'saved_to': str(args.out)
    }, indent=2))


if __name__ == '__main__':
    main()
