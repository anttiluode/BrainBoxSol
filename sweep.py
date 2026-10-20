"""Post-hoc stress sweep: all 16 sign patterns x 24 sequential write orders.

The 32 projected steps per task have the same nominal number of
response/sensitivity *calls* (320) as 80 regular steps per task (320):
projected uses one new task + all previously protected tasks per update.
SVD overhead and temporary memory are extra and NOT compute matched.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from brainboxsol import Medium, teacher_medium, write_one, write_protected


def run_sweep(steps=80, protected_steps=32):
    box = Medium()
    baseline = box.values(box.initial)
    patterns = list(itertools.product((-1, 1), repeat=box.blocks))
    orders = list(itertools.permutations(range(box.blocks)))
    rows = []
    for signs in patterns:
        target = box.values(teacher_medium(box, signs))
        for order in orders:
            trial = {}
            for method in ('local', 'global', 'protected'):
                state = box.initial.copy()
                learned = []
                for k in order:
                    count = protected_steps if method == 'protected' else steps
                    for _ in range(count):
                        if method == 'protected':
                            write_protected(box, state, k, target, baseline, learned)
                        else:
                            write_one(box, state, k, target, baseline, method)
                    learned.append(k)
                trial[method] = float(np.mean(abs((box.values(state) - target) / baseline)))
            trial['teacher_signs'] = list(signs)
            trial['write_order'] = list(order)
            rows.append(trial)
    arrays = {name: np.array([r[name] for r in rows])
              for name in ('local', 'global', 'protected')}
    return {
        'protocol': {
            'teacher_sign_patterns': len(patterns),
            'write_orders': len(orders),
            'total_cases': len(rows),
            'regular_steps_per_task': steps,
            'protected_steps_per_task': protected_steps,
            'regular_sensitivity_calls_per_case': box.blocks * steps,
            'protected_sensitivity_calls_per_case':
                protected_steps * (box.blocks * (box.blocks + 1) // 2),
            'not_matched': 'SVD compute, old-query replay and additional storage',
            'posthoc': True,
        },
        'mean_absolute_relative_error': {
            name: float(np.mean(array)) for name, array in arrays.items()},
        'median_absolute_relative_error': {
            name: float(np.median(array)) for name, array in arrays.items()},
        'local_beats_global_cases': int(np.sum(arrays['local'] < arrays['global'])),
        'protected_beats_local_cases': int(np.sum(arrays['protected'] < arrays['local'])),
        'protected_beats_global_cases': int(np.sum(arrays['protected'] < arrays['global'])),
        'direct_indexed_table_error': 0.,
        'qualifications': [
            'Same four fixed known query ports; old tasks are explicitly replayed.',
            'One deterministic medium, finite family of hand-designed teachers.',
            'No novel-learning, autonomous routing or PAC result.',
            'Projection is established orthogonal-gradient continual-learning machinery.',
            'Direct table stores all four scalar targets exactly at less cost.',
            'No future task family or independent dataset was held out.',
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('results/sweep.json'))
    args = p.parse_args()
    outcome = run_sweep()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(outcome, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(outcome, indent=2))


if __name__ == '__main__':
    main()
