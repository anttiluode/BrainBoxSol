# One-bit reward experiment

This extends the 81-node adaptive medium with a writer that receives binary action outcomes instead of an exact teacher gradient. It is a bounded two-cue association experiment, not evidence of a brain or general intelligence.

## Method and information boundary

The environment samples a cue, runs a proposed conductance field, samples an action from `sigmoid(5 * motor_score)`, and returns one bit for correctness. `RewardWriter` receives only those bits and its own parameter probes. It never receives a cue label, target action, motor score, or adjoint. With binary actions, action plus correctness logically reveals the target; this boundary removes an explicit teacher gradient, not the information implicit in feedback.

Each update freezes the current field, proposes eight plus/minus random perturbation pairs, and uses the same source and policy coin within each pair. Four pairs use each cue. Projected probe differences estimate a reward ascent direction. The updated 272 positive conductances remain within the original bounds and fixed sum. This is global parameter search with paired perturbations; it is not a local biological plasticity rule. Training actions are stochastic; evaluation uses the fixed greedy motor readout.

Reward, shuffled-reward, and random-update arms each use 512 updates, eight pairs, and 8,192 physical forward trials per mapping. Shuffled rewards permute the 16 actual bits within each batch, preserving the reward count. Random updates rotate the estimated update into a fresh random direction with the same norm **before** the budget projection; clipping can change the final norm. All three arms receive identical input sequences. Frozen and erased arms use the exact initial field. Twenty permutations of the learned field test spatial organization separately.

The teacher reference retains the original 600 updates: 1,200 forward and 1,200 adjoint solves per mapping. It is not compute matched. Costs include training forwards, adjoints, solver steps, wall time, persistent conductance storage, and named working-array payload estimates. Memory figures exclude interpreter overhead and temporary NumPy allocations. Evaluation time is separate. No efficiency advantage is claimed.

## Selection and locked evaluation

Development uses seeds 211, 223, 241. All exploratory settings and retained curves are in `results/reward_medium/development_pilots.json`; the initial exploratory run is recorded without a retained curve. One reversed-mapping seed was used for a four-setting sweep. The selected rate .01 and 512 updates were then checked on the other development seeds. `development.json` contains the complete selected-setting comparison against controls. Development was run from an uncommitted working tree; its `code_commit` names the base checkout, not a reproducible source snapshot. The fresh run uses the exact committed lock source. Subsequent development-time edits concerned grid-size plumbing, budget/replay wording, and error handling, not the default-grid learning calculation.

Development mean reward accuracy was 83.33% (seed means 100%, 75%, 75%), versus 50% shuffled reward, 50% random updates, 50% frozen/erased, and 100% teacher. All aggregate criteria passed on development; this is selection evidence, not a held-out success. The reversed mapping reached 100%, 50%, 50% across the development seeds.

The fresh evaluation uses seeds 301–310, both mappings, and 100 held-out examples per cue (4,000 primary decisions). The committed protocol must say `locked: true`, and full mode requires an exact clean lock commit. Settings must not change in response to these results. Full mode records its code commit and protocol hash; development and smoke results cannot report a scientific gate pass.

The gate requires mean accuracy at least 80%, at least 20 percentage points over frozen, shuffled reward, and random updates, erasure removing at least half the gain over frozen, no invalid trials, and a time-step refinement score difference below .02. Means average mappings within each seed, then seeds. This is an aggregate gate: it does **not** require every seed or mapping to succeed. Per-mapping results and seed ranges must accompany any pass claim. Spatial shuffling is diagnostic, not a required threshold.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -v
OPENBLAS_NUM_THREADS=1 python reward_gate.py --mode development --out results/reward_medium/development.json
# Use the recorded lock commit from full.json, in a clean checkout at that commit:
OPENBLAS_NUM_THREADS=1 python reward_gate.py --mode full --lock-commit COMMIT --out results/reward_medium/full.json --replay results/reward_medium/replay.html
```

The replay displays recorded Before/Learned/Frozen/Erased trajectories for both cues and mappings, not an animation used to infer scores. It illustrates the first fresh seed (301); the table averages all fresh seeds. The full receipt contains all arm materials, motor energies, curves, costs, and numerical checks.

## Result

Fresh held-out evaluation pending. This section will be replaced with the complete result, including negative findings.
