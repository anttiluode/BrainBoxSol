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

Raw development measurements are in `results/adaptive_medium/development.json`.

## Locked held-out result

**PASS, within the stated two-cue graph task.** The complete code and protocol
were committed as `032974c6269d54ebe24c1c3ec0e12ffa07028ea6` before any held-out
training. Seeds 101–110 each ran both mappings, with 100 fresh variants per cue:
4,000 learned-material decisions, all correct. No settings changed after
development and no held-out run was selected or retuned.

| Arm | Mean accuracy | Seed standard deviation | Seed range |
|---|---:|---:|---:|
| learned | 1.000000 | 0.000000 | 1.000000–1.000000 |
| frozen | 0.500000 | 0.000000 | 0.500000–0.500000 |
| erased | 0.500000 | 0.000000 | 0.500000–0.500000 |
| shuffled | 0.503275 | 0.002528 | 0.500000–0.507250 |
| activity_only | 0.500000 | 0.000000 | 0.500000–0.500000 |
| linear | 1.000000 | 0.000000 | 1.000000–1.000000 |
| lookup | 1.000000 | 0.000000 | 1.000000–1.000000 |

Accuracy, gain over frozen, erasure, shuffle sensitivity, episode validity and
numerical checks all passed their unchanged thresholds. No motor-energy
failures occurred in any material arm, including all 20 shuffle permutations.

| Seed | Learned | Frozen / erased / traffic | Shuffled | Linear / oracle |
|---|---:|---:|---:|---:|
| 101 | 1.000 | 0.500 | 0.505625 | 1.000 |
| 102 | 1.000 | 0.500 | 0.502000 | 1.000 |
| 103 | 1.000 | 0.500 | 0.500000 | 1.000 |
| 104 | 1.000 | 0.500 | 0.500000 | 1.000 |
| 105 | 1.000 | 0.500 | 0.507250 | 1.000 |
| 106 | 1.000 | 0.500 | 0.501250 | 1.000 |
| 107 | 1.000 | 0.500 | 0.502375 | 1.000 |
| 108 | 1.000 | 0.500 | 0.505375 | 1.000 |
| 109 | 1.000 | 0.500 | 0.502500 | 1.000 |
| 110 | 1.000 | 0.500 | 0.506375 | 1.000 |

Maximum time-refinement score difference: 0.002063; zero changed actions.
Minimum total motor energy across all evaluated material arms: 0.00632627.
The spot-check gradient error was 1.31e-10;
full-rollout/response-map energy discrepancy was 2.08e-17.

The full local run took 89.73 seconds, including all controls.
Mean supervised-medium training time per seed/mapping was 2.730 s
(range 2.206–3.544); the linear classifier averaged
0.0741 s (range 0.0570–0.1635). Timings are
local measurements, not a hardware-normalized benchmark.

The medium stores 272 float64 conductances (2,176 bytes), compared with six
linear-classifier parameters (48 bytes) or a two-value oracle (16 bytes).
One supervised training condition uses 1,200 forward and 1,200 adjoint
rollouts, 384,000 state steps, and 208,656 bytes of forward/adjoint trajectory
arrays at peak, excluding gradient temporaries. Evaluation builds an exact
two-motor response map for each frozen material; its work is reported separately.
The medium is much more expensive than the classifier on this task.

The aligned initial mapping was already 100% correct; the reversed initial
mapping was 0%. Both become 100% after supervised learning. Thus material
adaptation overcomes an initial geometric bias; this is not open-ended concept
learning, a newly discovered local learning law or a complete brain.

## Replay and reproducibility

Open [the self-contained replay](../results/adaptive_medium/replay.html) locally.
It uses seed 101 without best-seed selection. All activity frames are recorded
rollouts; playback does not train. The static preview uses actual conductances
and complete-window motor energies. Browser interaction/layout could not be
smoke-tested: Playwright was present but its browser executable was absent.
Embedded-data round-trip, escaping and JavaScript syntax checks passed.

Raw measured results: [full receipt](../results/adaptive_medium/full.json).
Per-case motor responses are retained for learned/frozen/erased/traffic arms.
Shuffled controls retain exact per-permutation accuracy, invalid counts and
motor-energy minimum/mean/maximum plus the deterministic permutation stream;
individual shuffled cases can be regenerated from the locked code.

To reproduce the original run, use a clean checkout of the lock commit:

```bash
git checkout 032974c6269d54ebe24c1c3ec0e12ffa07028ea6
OPENBLAS_NUM_THREADS=1 python adaptive_medium_gate.py --mode full --protocol protocols/adaptive_medium.json --lock-commit 032974c6269d54ebe24c1c3ec0e12ffa07028ea6 --out results/adaptive_medium/reproduced.json --replay results/adaptive_medium/reproduced.html
```

On Windows, omit the environment-variable prefix or set it separately. Full
mode rejects uncommitted tracked changes and a mismatched lock commit. An
interrupted full run can resume with `--resume` only from a matching receipt
and its exact code/protocol identity.

## Independent review and post-lock hardening

A fresh reviewer independently ran the 31 original tests and regenerated all
stored learned/frozen/traffic material responses across the ten seeds, the
summary/protocol digest and all replay episodes. Two Important validation
defects were found and fixed: resume now validates materials and regenerates
all decision evidence before accepting completed seeds; replay generation
validates every browser-consumed episode field and selector coverage.

The resume CLI regression fails against the original locked code; all 33 tests
pass after both fixes. The replay HTML wrapper was regenerated, while the
original `full.json` and its producing lock commit remain unchanged. Neither
fix changes fresh-run learning, task physics or the frozen measured result.
See [review record](adaptive_medium_review.md) for scope and limitations.
