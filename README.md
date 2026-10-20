# BrainBoxSol — content writes inside a spectral address scaffold

A small, inspectable experiment linking the 2024 SuperWeights question to the October 19, 2026 SihtiBrain return-path discussion.

**Hypothesis:** weak bridges between strongly coupled local blocks create slow collective modes. If those bridges define a relatively stable *address scaffold*, local material changes can store content without globally rewriting that scaffold. An exact local forward/adjoint sensitivity gives a supervised way to write the material. We then test whether changing the bridges damages the result.

**Status: exploratory CPU toy with one fixed teacher and one write order; not a demonstrated new AI architecture, a biological lipizone model, or a learned PAC code.** The results below are from a local run of the supplied deterministic code and can be regenerated. The gates were not preregistered before running.

**Proposed next subsystem:** [Experience changes the medium — approved adaptive material design](docs/superpowers/specs/2026-10-20-adaptive-medium-design.md). A separate fast signal and persistent conductance grid would learn two cue-to-output associations, with reset, erasure, shuffle and ordinary-classifier controls. The [implementation plan](docs/superpowers/plans/2026-10-20-adaptive-medium.md) is ready for review and execution-method selection; no implementation or result is claimed.

## Run

Requires Python 3.10+ and NumPy.

```sh
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python brainboxsol.py --out results/gate0.json
OPENBLAS_NUM_THREADS=1 python -m unittest -v test_brainboxsol
```

On Windows, omit the `OPENBLAS_NUM_THREADS=1` prefix or use `set OPENBLAS_NUM_THREADS=1` in Command Prompt.

## Physical toy

16 nodes are grouped into four four-node blocks. Each block has two internal paths (4 writable edges at conductance 0.9); adjacent blocks connect through three *frozen* weak bridges (conductance 0.055). Every node has leak 0.035. A unit current enters at a known block tip; the only response is one scalar at the first node (the soma). No recurrent neural network, transformer or oscillatory PAC mechanism is involved.

With incidence matrix `B`, edge conductances `g` and leak `ell`:

```text
A(g) = B diag(g) B^T + ell I
A v = b_k                 (forward response)
A z = c                   (adjoint from soma)
y_k = c^T v
dy_k/dg_e = -(B_e^T v) (B_e^T z)
```

The exact derivative uses the local *forward voltage drop* times the local *adjoint voltage drop*. That is not the same thing as a neutral unlabelled ping. To fit a scalar *supervised* teacher target, the writer also needs `(y_k-target_k)` and a global gradient-norm normalization. We do not claim that every piece of this update is biologically local.

The target is computed once from a hidden teacher with alternating within-block conductance multipliers `[1.4, 0.6, 1.4, 0.6]`. The student starts from the homogeneous medium. Tasks 0, 1, 2, 3 are trained sequentially for 80 updates each; after every task, we measure **all four** readouts. All fast voltage states are reset between queries.

Comparison:
- **Local:** only four conductances inside the queried block can change.
- **Global:** every 16 intra-block conductances can change; the same number of update steps and normalization rule.
- **Frozen:** no writes.
- **Direct indexed table:** four stored target scalars give exact responses on these four known queries, at vastly lower memory cost. This is an important *stronger* control, not an arm we beat.

## Exploratory receipt, 2026-10-20

| Measurement | Result |
| --- | ---: |
| Ratio first within-block eigenvalue / last macro eigenvalue | **22.6775** |
| Max absolute finite-difference error of adjoint sensitivity | **5.73e-7** |
| Final mean absolute relative error: local writes | **0.00155** |
| Final mean absolute relative error: global writes | **0.00872** |
| Final mean absolute relative error: frozen | **0.01989** |
| Direct four-value table error | **0** |
| Bridges modified during either training arm | **No** |

The local-write arm exhibits lower cross-task interference **in this selected four-task configuration**. Global has more writable parameters, not a matched number of activated parameters. The student gets a supplied query-port identity and a supervised target. Nothing here establishes autonomous addressing, extra storage capacity, transfer to unseen tasks, or superiority over a conventional memory.

### The bridge-drift negative result

After local learning, change the **three bridge conductances by −20%, +20%, −20%**, without mathematically transforming the rest of the medium as a gauge symmetry would require.

The four relative errors change from approximately

```text
before drift:   +0.0030  -0.0011  +0.0021  +0.0000
after drift:    +0.0402  -0.1467  -0.0000  -0.1457
```

Even when we supply *all four old targets* and perform another 160 local supervised update steps, two errors remain large:

```text
supervised replay: +0.0001  -0.1165  -0.0002  -0.1037
```

**The proposed homeostatic preservation gate fails in this toy.** Strongly constrained local material updates cannot reliably compensate for drift in the shared address bridges. An exact boundary-preserving coordinate transform would be a trivial different control; this is genuine function-changing drift.

## G1 — a 384-case correction and a protected-write control

**The fixed G0 example was misleading as a general comparison.** We expanded the study *after seeing G0* to all 16 binary teacher patterns (±40% within each block) × all 24 permutations of the four write tasks: **384 deterministic cases**, with the same bridge topology, learning rate and target family. This is a post-hoc stress screen, not independent generalization.

Run:

```sh
OPENBLAS_NUM_THREADS=1 python sweep.py --out results/sweep.json
```

| Strategy | Mean absolute relative readout error across 384 cases | Wins |
| --- | ---: | --- |
| Local-only writes, 80 updates/task | **0.015412** | Beats global in **141/384** cases |
| Global writes, 80 updates/task | **0.010981** | Beats local in **243/384** cases |
| Projected global writes, 32 updates/task | **0.000593** | Beats both in **384/384** cases |
| Four-value indexed lookup | **0** | Exact and cheaper |

The correction matters: simply protecting the bridge geometry and updating only the addressed block **does not reliably protect content**. Cross-region effects still occur through the shared network.

The third arm uses **prior-art orthogonal gradient projection** (OGD/GPM-style). The old readouts' sensitivity rows are computed by the same physical forward×adjoint edge formula. Let `J_old` stack those rows over the 16 writable edges; project the new proposed gradient `u` to

```text
u_safe = (I - J_old^T (J_old J_old^T)^+ J_old) u
```

The code uses an SVD row-space projection. This suppresses new updates in directions to which **previously learned outputs** are first-order sensitive. Because the Jacobian is recomputed every step, the protection is more robust to local nonlinearity than projecting just once.

**Cost boundary:** the global and local arms use 4 × 80 = 320 forward/adjoint sensitivity evaluations; the protected arm uses 32 × (1+2+3+4) = **320** evaluations by explicitly re-querying all earlier tasks. However, its SVDs, old-task replay, data/control requirements and computation overhead are *not matched*. It also has access to the list of old task ports. This is not a free biological learning rule or autonomous indexing. It is not a fair efficiency win over direct storage. No held-out task family was evaluated.

The protected arm has **not** passed the bridge-drift/homeostasis gate: G0's drift failure remains, and projection cannot magically make a physically changed address scaffold invariant.

## G2 — Phase/address synthesis under actual bridge drift (2026-10-20)

New **post-hoc** combination of BrainBoxClaude's G2/G3 ideas with this repository's protected-material writer: [full gate report](docs/phase_bridge_synthesis.md).

**Key finding:** Current-phase timing improves access in an unprotected medium, but does **not itself protect** the memory against bridge drift. Conserving each block-to-block interface's total conductance protects both old response functions and phase addresses in this selected graph. Fixed externally assigned time slots are a strong control that can outperform PAC-like release after drift. Not a new learning method; no RNN comparison yet.

Run: `python phase_bridge_gate.py` and `python -m unittest -v test_phase_bridge_gate`.

## G3 — the next genuine discriminator (not yet run)

Take the protected writer into a task with more potential experiences than explicit readout slots. Freeze hyperparameters, constrain the old-query replay budget, add context-gated readouts and physically perturbed bridges, and compare with capacity-/compute-matched conventional gradient projection, recurrent memory and direct indexed memory. A geometry-mediated benefit must survive those controls.

## G3 — Outer response lens: the hidden material speaks through time

**[Full protocol, step-response diagram, result table and limitations](OUTER_LENS.md)**

A material box can have two distinct internal conductance arrangements with **identical static DC response and identical immediate response**, but different transient behavior. Two parallel hidden capacitive paths encode externally imposed A/B histories; a parent reader observes only boundary input/output pairs.

A three- or five-coefficient classical ARX reader learns the externally visible dynamics and predicts new temporal drives. A matched-size FIR reader and one-scalar block mean/DC model fail to predict the long transients. The two-pole reader fits exact data but becomes ill-conditioned when training measurements are noisy. Across five probe seeds and three heldout drive families, the *noisy* B-case test RMSE is roughly 0.01075 for ARX-2 versus 0.22717 for FIR-5 and 0.46525 for static DC.

**Scope:** this tests *how to describe the inside from outside*, not a learned writing rule, a new numerical method, PAC, PhiWorld intelligence or a better AI. It is complementary to Claude's G6 continual-learning branch.

Reproduce: `python -m unittest -v test_outer_lens && python sweep_outer_lens.py`.

## What about PAC as an address?

Phase–amplitude coupling (PAC) could provide a *time-selective read gate* across hierarchical regions: a slow phase might determine which region's fast activity is read in each temporal window. But the present medium is passive, with **real exponential decays, not oscillations**. A slow Laplacian eigenmode is not automatically a phase oscillator.

PAC would require an added dynamical mechanism, an unambiguous phase assignment, nonlinear phase-dependent gating and a comparison against ordinary timed multiplexing. Simultaneously active sites otherwise superpose at the soma. Equal phases could collide; a phase label is not automatically a unique address.

**Separate next gate (not implemented):** give blocks controlled slow oscillations, use known balanced simultaneous inputs, and compare phase-gated versus ungated readout at equal bandwidth, probe cost, and noise. Include a predetermined time-slot control and a nonhierarchical oscillator control. Only claim PAC addressing if it wins after those controls.

## Connections and next discriminator

- [SihtiBrain](https://github.com/anttiluode/SihtiBrain): hidden branch distinctions and state-dependent return.
- [GrowingMatrix](https://github.com/anttiluode/GrowingMatrix): local voltage-drop sensitivities and material-constrained optimization.
- [GraphNN](https://github.com/anttiluode/GraphNN): write/selectivity and direct-address comparisons.
- [DaFuc](https://github.com/anttiluode/DaFuc): the original dynamic SuperWeights ambition.
- [Genealogy: Mad Monday](https://github.com/anttiluode/Genealogy/blob/main/docs/reviews/2026-10-19-mad-monday.md): why the return must interact with the actual medium, not just with its final output.

Claude's separate project is testing **spectral/PAC addressing**. BrainBoxSol focuses on **local content writes, selective interference, and the fragility of geometry-derived addresses under real drift**.

The next worthwhile experiment is *not* a larger network. It is a frozen prospective evaluation across fresh target patterns and write orders, matched effective writable degrees of freedom, and a replay-free local compensator under independent bridge perturbations. It must beat strong direct indexing or an equal-budget fast-weight/recurrent control on a problem where those methods are allowed to operate normally.

## Closest established mechanisms (explicit prior art)

- [Farajtabar et al. (2020), Orthogonal Gradient Descent](https://proceedings.mlr.press/v108/farajtabar20a.html): project future updates away from directions that change old outputs.
- [Saha et al. (2021), Gradient Projection Memory](https://openreview.net/forum?id=3AOj0RCNC2): use an SVD-based protected subspace for continual learning.
- [Lisman and Jensen (2013), The Theta-Gamma Neural Code](https://pmc.ncbi.nlm.nih.gov/articles/PMC3648857/): phase/subcycle organization of multiple items, motivating but not proving PAC as a robust spatial address.

BrainBoxSol's limited implementation tests an established projection idea against a concrete local conductance response Jacobian. It does not establish novelty of the projection, nor a biological PAC mechanism.
