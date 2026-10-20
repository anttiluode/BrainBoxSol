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

**The locked gate passed.** Jointly matched temporal currents gave **1,279/1,280 correct (99.9219%)** across ten fresh simulated devices. Fixed pulses and erased material gave 50%, and shuffled waveform assignment gave 50.7031%. No learning, matching, port, or verdict settings changed after the lock commit `0d45a0c35d826c798ae9bd7872e49b664b3e5d9f`.

| Arm | Mean accuracy | Device range |
|---|---:|---:|
| joint | 99.9219% | 99.2188%–100.0000% |
| raw | 100.0000% | 100.0000%–100.0000% |
| energy_only | 99.9219% | 99.2188%–100.0000% |
| fixed | 50.0000% | 50.0000%–50.0000% |
| shuffled | 50.7031% | 45.3125%–57.0312% |
| erased | 50.0000% | 50.0000%–50.0000% |
| charge_only | 100.0000% | 100.0000%–100.0000% |
| peak_scalar | 96.6406% | 94.5312%–98.4375% |

| Device seed | Joint waveform | Shuffled waveform | Fixed / erased |
|---|---:|---:|---:|
| 501 | 100.0000% | 45.3125% | 50% / 50% |
| 502 | 100.0000% | 51.5625% | 50% / 50% |
| 503 | 100.0000% | 54.6875% | 50% / 50% |
| 504 | 99.2188% | 57.0312% | 50% / 50% |
| 505 | 100.0000% | 46.8750% | 50% / 50% |
| 506 | 100.0000% | 46.8750% | 50% / 50% |
| 507 | 100.0000% | 53.1250% | 50% / 50% |
| 508 | 100.0000% | 48.4375% | 50% / 50% |
| 509 | 100.0000% | 51.5625% | 50% / 50% |
| 510 | 100.0000% | 51.5625% | 50% / 50% |

The primary errors came from seed 504 (127/128 correct); the other nine devices scored 128/128. The maximum charge error was 3.55e-15, the maximum squared-current-integral error was 7.11e-15, and the largest step-halving receiver-voltage difference was 0.00180905. The audit exactly recomputed all 1,280 primary receiver voltages and decisions, refit all stored receiver thresholds from training observations, verified control predictions and shuffle permutations, regenerated example written materials/probe traces, and matched protocol/source hashes to the lock commit.

This establishes a limited causal chain: activity changes persistent material; after fast activity is erased, identical probes produce distinguishable temporal responses; those responses cause distinguishable states in another frozen medium. Erasing the material or breaking history/waveform assignment removes reliable decoding.

It does **not** establish a need for the full waveform: raw total charge alone scored 100%, and a scalar peak of the jointly matched waveform scored 96.6406%. Our charge/energy matching rules exclude two particular scalar explanations, not every possible scalar representation. The second RC box itself is a temporal linear filter whose one scalar observation is read with supervised calibration. This experiment does not repair or rerun the earlier reward-learning failure.

[Mechanism figure](../results/history_waveform/mechanism.png) and [all-device chart](../results/history_waveform/results.png) show actual recorded quantities. [Standalone visual report](../results/history_waveform/report.html). The complete receipt is `results/history_waveform/full.json`; the audit is `audit.json`. All 49 unit tests pass.
