# Adaptive material memory

The first subsystem separates fast diffusing activity from a persistent
conductance grid. A supervised transient adjoint changes the conductances;
fixed motor regions read the response to a later sensory cue. Every episode
starts with zero activity, and material is frozen during evaluation.

## Development evidence (not held out)

Seeds 11, 23 and 41, both cue-to-motor mappings, 600 updates, 100 fresh examples
per cue, 20 shuffled-material controls. All initial design settings were kept;
no hyperparameter search or threshold changes were needed.

| Arm | Mean development accuracy |
|---|---:|
| Learned material | 1.000 |
| Frozen / erased material | 0.500 |
| Shuffled learned material | 0.511 |
| Traffic-only reinforcement | 0.500 |
| Linear classifier, six parameters | 1.000 |
| Oracle lookup, given cue identity | 1.000 |

Report the two mappings separately: the initial geometry already solves the
aligned mapping (1.000), while it gets the reversed mapping wrong (0.000).
Supervised material learning solves both. Shuffling the trained reversed
material removes almost all of that change. The mean therefore tests learning
against an explicit initial spatial bias, not discovery of arbitrary concepts.

The gradient finite-difference and exact response-map checks pass. Halving dt
at the same physical time changes development scores by less than 0.002 and
changes no tested actions. These development cases satisfy the proposed
criteria, but `scientific_gate_passed` remains false until the locked full run.

## Numerical and conceptual scope

The exact fixed-material motor response map accelerates evaluation only; tests
compare it with the entire activity rollout. The writer stores a trajectory
and computes a reverse-time adjoint from teacher labels. It is supervised
reverse-mode differentiation, not an autonomous biological return signal.

The graph has 81 nodes and 272 pre-existing candidate edges. Only positive
conductance values change, with a constant total budget. Thresholded edges in
the replay are a visualization, not literal addition or deletion of wires.

The conventional classifier uses six trainable numbers, compared with 272
conductances. This tiny association task is expected to favour ordinary digital
learning in cost. A successful material gate establishes a causal mechanism,
not a performance advantage, continual learning or a complete brain.

## Run

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -v
OPENBLAS_NUM_THREADS=1 python adaptive_medium_gate.py --mode smoke --out results/adaptive_medium/smoke.json
OPENBLAS_NUM_THREADS=1 python adaptive_medium_gate.py --mode development --out results/adaptive_medium/development.json
```

The held-out command and result will be added after the complete implementation
is committed and locked. Raw development measurements are in
`results/adaptive_medium/development.json`.
