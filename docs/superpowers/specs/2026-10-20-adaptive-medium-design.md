# Experience changes the medium: first adaptive BrainBoxSol subsystem

Date: 2026-10-20 (Europe/Helsinki)

Status: design for user review. No adaptive-medium implementation or results
exist in this change. Conversational scope was approved by the instruction to
continue in BrainBoxSol; the written specification is the next review point.

## Intended outcome

Build a small, inspectable system in which sensory activity passes through a
material network, supervised feedback changes that material, and the retained
change improves later responses after all fast activity has been cleared.
The user wants a concrete step toward dynamic AI and eventually a brain-like
organism, motivated by PhiWorld and two years of SuperWeights experiments.

The first deliverable demonstrates a causal write-retain-use loop on two
cue-to-output associations. It must show the evolving connections and their
effect on behaviour, with reproducible measurements and negative controls.
It is not a claim of a complete brain or a competitive new AI algorithm.

Assumptions: CPU-scale NumPy implementation; fixed sensory and motor interfaces;
an explicit supervised teacher for the first writer; no modification of
BrainBoxClaude. Existing G0-G3 experiments retain their current behaviour.

## Current repository and integration

The starting point is commit
`852eed3079b3c101c3056c35969e19a13b7dc3b0` on
`sol/outer-response-lens-g3`, the head of PR #3. That PR is stacked on #2 and #1;
`main` currently contains only the license. This design gets a separate branch
and a draft PR targeting the G3 branch. No earlier PR is merged by this work.

`brainboxsol.py` already provides positive conductance matrices, exact static
forward/adjoint sensitivities and cleared-state impulse responses. Its graph is
hand-blocked and its teacher supplies response values. `outer_lens.py` identifies
externally imposed material histories from boundary transients; it does not
learn the material writes. The new subsystem uses the same operator convention
but adds a spatial graph, transient supervised writes and fixed motor outputs.

## Approaches considered

| Approach | Benefit | Main limitation | Decision |
|---|---|---|---|
| Spatial conductance graph with separate fast activity | Positive operator, explicit material budget, exact discrete adjoint, easy ablations | Fixed candidate connections; no spontaneous neurites | First build |
| Continuum field with slow anisotropic conductivity | Closest to the proposed PhiWorld highways | Tensor discretization, pinning and numerical convergence require separate validation | Later bridge to PhiWorld |
| Add reinforcement directly to the existing scalar PhiWorld field | Small apparent code change | Structure and travelling signal remain confounded; stability is not useful memory | Do not use for this gate |

A two-phase scalar wall is not automatically an axon or a branching network.
Likewise, a slow Laplacian mode does not guarantee a unique readable address.
Those are later hypotheses, not prerequisites to declare satisfied here.

## State and operator

Use a 9 by 9 open-boundary grid: 81 activity nodes, with horizontal, vertical
and both diagonal candidate edges between neighbours. There is no periodic
wrap. The diagonal crossings are graph edges without an extra shared junction;
this abstract graph is not a physical two-dimensional sheet.

Let B be its oriented node-by-edge incidence matrix and g its persistent edge
conductances. Fast activity u is separate from g:

```math
A(g) = B\,\mathrm{diag}(g)B^T + \ell I,
\qquad u_{t+1} = u_t + \Delta t\,[s_t-A(g)u_t].
```

All conductances are positive. Initial numerical settings are leak 0.02,
time step 0.1, initial mean conductance 0.08, minimum 0.005 and maximum 0.4.
For maximum weighted degree d_max, require
`dt * (2*d_max + leak) <= 0.8` before a run. This conservative bound makes
the unforced update contractive. Reject invalid settings instead of silently
changing the protocol. These defaults satisfy the bound even at the edge cap.

The material feasible set is

```math
g_{\min}\leq g_e\leq g_{\max},\qquad
\sum_e g_e=|E|g_0.
```

Initialize with seeded ten-percent perturbations around g0 and project to this
set. Training also uses Euclidean projection to the bounded simplex, implemented
as `clip(candidate - threshold, lower, upper)` with a scalar threshold chosen
to meet the sum. Clipping followed by rescaling is not an equivalent projection.
Require bounds within 1e-12 and total-budget error at most 1e-10 times the
budget. Failure to reach this tolerance is a numerical error.

Conductance redistribution is the first definition of structural adaptation.
The graph never adds or deletes candidate edges. Any thresholded image must
state its threshold and must not be described as literal topology growth.
No independent wall field, myelin variable or learned probe policy is included.

## Sensory task and fixed readout

Coordinates below are zero-based (row, column). Cue A occupies a 3 by 3 patch
centred at (2, 1); cue B uses (6, 1). Each source is a nonnegative normalized
Gaussian within its patch, width 0.7 grid units. The two motor regions are the
three nodes at column 7 and rows 1-3 or 5-7. These interfaces never move or learn.

Each episode starts with exactly zero activity and lasts 160 time steps. Apply
the sensory source for the first 24 steps. Integrate the two motor regions over
states 113 through 160, obtaining nonnegative responses E0 and E1. The fixed
decision score is `r = (E0-E1)/(E0+E1+1e-9)`; choose motor 0 if r is positive,
motor 1 if negative, and record an exact tie as incorrect. Store raw response
energies as well as the ratio so tiny-signal artifacts remain visible.

The teacher assigns A and B to opposite motors. Evaluate both possible
assignments for every initialization seed. Their labels enter only the training
loss; test simulation and the motor readout receive no task identity or target.

Training episodes use source amplitude uniform in [0.8, 1.2], centre jitter
uniform in [-0.25, 0.25] on each axis, and independent per-patch additive noise
with standard deviation 0.05 times the unperturbed patch maximum. Clip the
patch to nonnegative values, normalize its sum to one, then apply amplitude.
Evaluation uses fresh random draws, centre jitter [-0.35, 0.35] and noise 0.10.
These are new variants of two known cues, not unseen categories or tasks.

Test labels are held by the evaluator and never passed to an acting component.
Clear u before every training and evaluation episode. Freeze g during all
evaluation episodes. Clearing fast state happens before presenting the new
cue; this task does not demand an answer about an erased cue in a blank system.

## Teacher return and material update

For target sign z in {-1, +1}, use loss `0.5*(r - 0.8*z)**2`. The teacher supplies
z, not a desired internal path. A minibatch contains one independently perturbed
episode of each cue; average their material gradients before one update.

Compute the exact gradient of the finite discrete rollout. If q_t is the direct
loss derivative with respect to state u_t and M = I - dt*A(g), the adjoint is

```math
\lambda_T=q_T,\qquad
\lambda_t=q_t+M^T\lambda_{t+1},\qquad
\frac{\partial L}{\partial g_e}
=-\Delta t\sum_{t=0}^{T-1}(B_e^Tu_t)(B_e^T\lambda_{t+1}).
```

The q terms include the full derivative of the motor energy ratio. Forward
state and adjoint differences yield an edge-local product, but obtaining the
adjoint requires stored or recomputed trajectories, the teacher loss and a
backward computation. This is supervised reverse-mode differentiation through
a symmetric medium, not a demonstrated biological ping or autonomous credit
assignment. Charge the trajectory memory and backward work in the report.

Start development with learning rate 0.05 and 600 minibatch updates. Use the
un-normalized gradient and bounded-simplex projection. A projected proposal
that contains nonfinite values is an error, not a value to clip away.

Development seeds are 11, 23 and 41. Numerical settings and training budgets may
be revised only on these seeds, with changes documented. Commit a complete
protocol lock, including evaluation code and constants, before running seeds
101 through 110. Do not quietly adjust thresholds after seeing those results.
This design alone is not a preregistered successful experiment.

## Comparisons and causal interventions

Run 100 fresh evaluation examples per cue, per mapping and per held-out seed.
Use identical evaluation inputs across compared arms. The primary unit of
replication is the initialization seed; average its two mappings before
reporting across seeds. Report every seed, mean, standard deviation and range.

| Arm | What it establishes |
|---|---|
| Learned material, fixed motors | Candidate mechanism |
| Frozen initial material, fixed motors | Benefit over its original physics |
| Erased material: restore the exact initial g | Causal necessity of retained changes |
| Shuffled learned g, 20 seeded edge permutations | Importance of the spatial arrangement at equal budget and histogram |
| Activity-only writer | Whether reinforcing traffic alone explains the benefit |
| Ordinary two-input linear classifier | Strong cheap digital control for this simple task |
| Direct two-entry lookup | Exact clean-identity reference, explicitly given cue identity |

For the activity-only arm, replace the supervised gradient by the positive
time-average squared voltage drop on each edge; project to the same feasible
set after each update. Use the same episodes and update count. This is a
traffic-reinforcement comparison, not a compute-matched gradient calculation.

The conventional classifier receives only the summed activity in the two
sensory patches (before propagation) and the same training labels. Train its
two-output linear map and biases by softmax cross-entropy on the same
minibatches, with learning rate 0.05 and the same episode count. Count its weights, compute
and persistent bytes separately. Lookup gets true cue identity and is an
oracle reference; do not imply it solves recognition without that information.
Neither control is expected to be beaten for efficiency on this tiny task.

## Verdict rules and failure handling

The useful persistent-material gate passes only if all of the following hold
on the locked held-out protocol:

1. Mean learned accuracy is at least 90 percent, with no seed below 80 percent.
2. Mean learned accuracy exceeds frozen accuracy by at least 20 percentage
   points; erasure removes at least half of that aggregate improvement.
3. Mean learned accuracy exceeds mean shuffled accuracy by at least 10 points.
4. All evaluated episodes have cleared initial activity, frozen evaluation
   material, finite state and strictly positive total motor response above
   1e-8. Count lower-energy episodes as failures, never omit them.
5. Material bounds, total budget, stability bound and numerical checks pass.

Failure of any conjunct is a failed gate. Publish all arms and mappings even
if the gate fails. A successful gate establishes supervised material memory
in this graph family, not greater capacity, better digital AI, continual
learning, spontaneous dendrites or a complete brain. Compare to the
activity-only arm descriptively; do not invent an autonomous-learning claim.

There is no interface-total conservation in this first grid: blocks and their
addresses have not been identified. A global material budget is a different
constraint and must be named separately. There is no protected old-task replay,
spectral address decoder or outer-lens predictor in the action path.

## Numerical verification and reviewable outputs

Before full evaluation, verify operator symmetry/positivity, conservation of
total activity for the zero-leak unforced case, dissipation with leak, absence
of wraparound and projection feasibility. Check the transient loss gradient
against centred finite differences on a small graph and several edges, away
from clipping boundaries. Repeat fixed-material rollouts at dt/2 with doubled
step counts and identical physical source and observation windows; report
score differences and any changed actions. Require maximum score difference
below 0.02 on development cases before protocol lock.

Keep implementation units separate: `adaptive_medium.py` for graph, rollout,
adjoint and projection; `adaptive_medium_gate.py` for training, controls and
receipts; `test_adaptive_medium.py` for meaningful numerical and reset tests;
`docs/adaptive_medium.md` for the resulting protocol and honest verdict.
Use the repository's NumPy and unittest conventions. Add the gate to CI with
a small smoke configuration; keep the full held-out run separately reproducible.

Produce a self-contained local HTML replay showing learned connections, fast
activity and motor responses for paired before/after episodes. Bundle sampled
data into the file so it requires no server or CDN. Label it as a replay, show
the seed, mapping and material budget, and include frozen/erased comparisons.
Use canvas or SVG for precise simulation data, with a static readable result
table if scripting is disabled. It is an inspectable experiment view, not a
claim that the replay itself performs training in the browser.

Raw JSON receipts include protocol version and git commit, seeds, all constants,
initial/final conductances, per-seed and per-arm metrics, intervention seeds,
raw motor-energy summaries, gradient/convergence checks, training curves,
evaluation failures, training/evaluation wall time, solver step counts,
persistent scalar counts and peak trajectory storage. Saving and reloading g
must reproduce evaluation scores on a recorded deterministic episode.

## Later decisions, contingent on this gate

Only after this mechanism is measured should we decide whether to introduce
PhiWorld-generated material priors, a separate structural order parameter,
anisotropic continuum transport, physically generated returns, reward-based
credit or active information-seeking probes. Each changes a different claim.
The next design should follow the measured failure or limitation, rather than
adding all of these ingredients at once.

## Review checkpoint

User review of this written specification comes before an implementation plan.
After approval, write the implementation plan and agree its execution method;
then implement and publish the measured result in BrainBoxSol. No G4 result is
claimed or implied by this design PR.
