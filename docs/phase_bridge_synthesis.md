# G4: Phase readout × learned content × bridge homeostasis

This is Sol's **post-hoc integration experiment** suggested by Gemini after
[BrainBoxClaude G2/G3](https://github.com/anttiluode/BrainBoxClaude)
and [BrainBoxSol G0/G1](../README.md). **It does not modify Claude's repo.**

## Hypothesis and split between mechanisms

- **Content:** sequentially fitted local *material conductances*, using
  BrainBoxSol's supervised OGD/adjoint-projected writer. This is a
  known-ports, known-targets toy, not unsupervised learning.
- **Address:** hierarchical chain/bridge geometry produces macro-block phases
  under a unit sinusoidal drive injected at the soma.
- **Reading:** a *prescribed* nonlinear fast release when the slow carrier
  reaches its local phase maximum (PAC-*like*, not a biological PAC simulation).
- **Homeostasis:** rescale parallel weak bridges inside each interface to
  preserve their **pre-drift summed conductance**. This is a stored local
  reference per interface. It is not inferred from the output.

The question is whether the combination protects information under drift,
not merely whether one can identify where a pulse occurred.

## Reproduce

```sh
pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python -m unittest -v test_phase_bridge_gate
OPENBLAS_NUM_THREADS=1 python phase_bridge_gate.py --out results/phase_bridge.json
```

Six macro-blocks of four cells, five in-block edges each, and **three
independent weak parallel bridge edges per adjacent pair** (15 bridge edges
total). All nodes have leak conductance. Five seeded graph variations; each seed
also chooses a distinct ±40% teacher content pattern and write order. The
writer takes 24 projected-gradient updates per known input port and stores the
result as the actual conductances.

After learning, apply independent bridge multipliers `exp(s*N(0,1))`
with `s=0.6` or `1.0`. Test four structural interventions:

1. No compensation.
2. Preserve the grand total conductance across **all** bridges.
3. Preserve **each individual interface** total (the candidate).
4. Standard symmetrical per-node degree scaling (the control).

All receivers use the same *pre-drift* response templates: NO decoder retraining.
Read 2 unknown-amplitude simultaneously active macro-blocks out of 6, with
1.2% noise relative to the template RMS, 50 paired random trials per seed.
The solver *is told K=2* and checks all 15 possible pairs, fitting two
nonnegative coefficients by clipped least-squares. Read window is identical
for every strategy: one cycle of the pacemaker plus a fixed tail.

Four read schedules:

- **Linear:** simultaneously release from two chosen sources at t=0.
- **Current-phase:** release at each location's current local slow-phase
  peak. Phases can drift; this is the self-timed PAC-like arm.
- **Old-phase:** use a permanently stored old phase schedule; NOT autonomous.
- **Fixed-slot:** externally assigned evenly spaced times per block;
  requires prior positional identity, and constitutes a strong baseline.

## Exploratory five-seed results

Source-pair exact-set recovery, where **higher** is better:

| Bridge drift s | Structural control | Linear | Current-phase | Old-phase | Fixed-slot |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.6 | None | .752 | .860 | .948 | **1.000** |
| 0.6 | Per-node | .780 | .884 | .964 | **1.000** |
| 0.6 | Interface-total | **1.000** | **1.000** | **1.000** | **1.000** |
| 1.0 | None | .392 | .504 | .728 | **.872** |
| 1.0 | Per-node | .436 | .556 | .808 | **.928** |
| 1.0 | Global-total | .412 | .468 | .696 | **.868** |
| 1.0 | Interface-total | .996 | **1.000** | **1.000** | **1.000** |

The post-drift SOMA transfer functions must remain close to the pre-drift
learned ones for **actual memory preservation**, not merely for decoding.
With `s=1.0`, mean absolute *relative* stored-response error:

| No homeostat | Grand-total | Per-node | Per-interface |
|---:|---:|---:|---:|
| **0.0616** | 0.1089 | 0.0552 | **0.00209** |

Mean phase displacement at the six ports: **12.52°** without compensation,
**18.08°** with one global bridge sum, **11.02°** with per-node scaling, and
**0.49°** with per-interface normalization.

Amplitude estimation confirms the result is not just exact-set thresholding:
at `s=1.0`, mean relative reconstruction error with current-phase read
is **0.625** unprotected vs **0.014** with per-interface normalization.
External fixed timing without a homeostat scores **0.271**, so
PAC-like timing is NOT the best possible addressing protocol here.

## Interpretation

**Pass, narrow:** In this graph family, preserving the integrated
conductance of each interface protects BOTH trained steady-state outputs and
the slow-phase address geometry through the tested bridge drifts. This is
essentially a coarse-grained parameter-conservation mechanism.

**No independent PAC protection:** Current-phase release modestly
improves two-source identification over a simultaneous linear read, but
**does not stop conductivity drift from altering the trained response**.
It may also shift pulse timing away from a stale decoder. Old-phase gating
and conventional fixed timing can be substantially more robust.

**Limits:** The code is an intentionally simplified 24-node numerical medium
rather than a reproduction of Claude's larger graph. The bridge structure
uses parallel interfaces whose aggregate naturally dominates coarse-scale
behavior; that is an inductive bias favoring the tested invariant. The
homeostat has explicit access to pre-drift interface totals, and the
supervised writer requires old targets, old-query replay and SVDs. The
decoder knows K and every old port template; the fixed-timing comparator
needs an externally assigned block code. Neither learning efficiency nor
capacity is benchmarked against a fair real RNN or indexed memory.

**Genuinely next:** freeze this protocol and test on a larger network
with *multiple distinct memories per block*, independent drifting changes
in **in-block** conductances, and a limited phase/inference budget; compare
with a fixed-slot RNN/fast-weight baseline using the same query information.
Also remove the oracle pre-drift interface totals and investigate whether
local feedback alone can maintain the bridge sums.

Do not claim that an imposed PAC-like gate implements a neural phase code,
that lipizones are electrical tensors, or that interface normalization is
a new method. This is a controlled synthesis of proposed mechanisms.
