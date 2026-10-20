# History transmitted through a temporal response

Question: after different histories and the same present probe, can a frozen second RC medium distinguish the first medium's history from a single temporal current port, even when charge, squared-current integral, onset, duration, and pulse count are fixed?

## Boundary and memory

Box 1 is the existing 81-node, 272-conductance positive diffusion grid. Histories stimulate its upper-left or lower-left patch with three pulses. Each activity-only update reinforces mean squared edge voltage drops, normalized by their spatial mean, followed by the original bounded fixed-budget projection. The writer has no label, target, reward, adjoint, or receiver measurement. Historical stimulation locations define the task classes; the spatial arrangement is designed, not discovered.

Every history starts with the same initial conductance field for its device seed. Each history pulse starts with zero activity. After writing, the probe again starts with zero activity, retaining only the modified conductances. The same single-site pulse at row 2, column 1 is applied to every case, and voltage at row 2, column 3 is recorded for 160 steps. Resetting the material to its initial field is the erased control. Thus the primary path excludes unfinished fast activity; it does not identify a biological channel-memory mechanism.

Box 2 is another frozen 81-node diffusion grid, initialized independently. Only the sampled current waveform enters its center node. Its final center voltage is the observation. Neither history label nor Box 1's conductances enter Box 2. A two-parameter threshold and orientation, calibrated on separate labeled training histories for each device/arm, turn that voltage into a decision. This calibration is explicit supervised readout; the second medium does not learn its own objective. A passive linear receiver cannot add information to its input.

Persistent payload: Box 1 has 272 float64 conductances (2,176 bytes); Box 2 has another 272 (2,176 bytes); the decision rule has two numeric parameters (16 bytes). Each physical box has 81 fast state values. The offline transport buffer has 160 float64 values (1,280 bytes). Array payloads exclude Python overhead and batch/analysis allocations. Devices, training histories, test histories, and calibration must not be described as one zero-memory channel.

## Matching and controls

Paired histories share initial material and all nuisance draws except stimulation location. Training and test use separate RNG streams; test histories also use the original task's larger jitter/noise. Acquisition noise is 2% of each natural trace's temporal standard deviation, with paired draws. It is added before waveform conditioning.

The primary conditioner acts on the whole captured trace:

`matched = 1 + 0.15 * (trace - mean(trace)) / std(trace)`.

This preserves centered temporal shape while adding a common DC pedestal. It is an external offline operation with a counted buffer and window latency, not a naturally emitted action potential or an online normalization law. Constant, nonfinite, or nonpositive-result cases are rejected. Every primary waveform is one positive pulse occupying the same 16-time-unit window, with charge 16 and squared-current integral 16.36. This current statistic is called "energy" for matching; actual Joule dissipation in the RC network is not equated with it.

| Arm | Current sent to Box 2 |
|---|---|
| joint | History response with charge and squared-current integral both matched |
| raw | Noisy natural response, without matching |
| energy_only | Natural response rescaled to a common squared-current integral |
| fixed | One template, fitted to the balanced training-trace mean, for every history |
| shuffled | Matched responses randomly reassigned among histories, independently in training/test |
| erased | Response after restoring Box 1's initial material, then jointly matched |
| charge_only | Raw total charge encoded as the amplitude of a fixed template |
| peak_scalar | Direct threshold on the matched waveform peak; bypasses Box 2 as a diagnostic |

The fixed template has the same charge/energy/support as the primary pulses. Shuffling preserves the waveform multiset. Erased cases are paired identical signals, including noise. The charge-only arm deliberately exposes raw charge information; it is not jointly matched. The scalar peak diagnostic prevents a claim that every scalar representation is blind. Equal charge and squared-current integral do not imply equal peaks or equal response to a temporal filter.

## Selection and verdict

One initial exploratory seed (401, 16 training pairs and 32 test pairs) gave joint 100%, erased 50%. No learning or matching settings changed. Complete development uses seeds 401, 409, 419, 32 training pairs and 64 test pairs per seed. All three reached 100% joint, fixed/erased 50%; shuffled ranged 46.875%–57.8125%. Raw charge and matched peak were also highly informative. The development receipt records source hashes because it was generated from an uncommitted working tree.

The full protocol is committed before untouched device seeds 501–510 are evaluated. Each device gets 64 labeled calibration examples and 128 held-out examples, for 1,280 primary test decisions in total. This tests new histories and newly calibrated device instances, not transfer of one fixed decoder across devices or unseen history categories.

The aggregate gate requires joint accuracy >=80%, at least 20 percentage points over fixed/shuffled/erased, each of those controls <=60%, charge/energy errors below 1e-10 with common positive support, and an absolute receiver-voltage difference below .02 under step halving. Numerical refinement holds the written material fixed and checks the first noiseless example of each history at both boxes' resolutions; it is not convergence verification of the writing process or every trial.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -v
OPENBLAS_NUM_THREADS=1 python waveform_gate.py --mode development --out results/history_waveform/development.json
# Exact clean tracked checkout at the lock commit recorded in full.json:
OPENBLAS_NUM_THREADS=1 python waveform_gate.py --mode full --lock-commit COMMIT --out results/history_waveform/full.json
```

Full mode verifies the committed protocol and implementation at the exact current commit. Smoke/development never claim held-out scientific success. The full receipt contains primary test waveforms, receiver material, decisions, calibration models, invariants, source hashes, example Box 1 materials and natural traces, solve counts, and refinement results.

## Fresh result

Pending locked evaluation. Any pass establishes a limited engineered history-transmission mechanism. It would not establish spikes, biological waveform coding, a teacher-free learning gradient, holography, or a brain.
