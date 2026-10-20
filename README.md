# BrainBoxSol — content writes inside a spectral address scaffold

A small, inspectable experiment linking the 2024 SuperWeights question to the October 19, 2026 SihtiBrain return-path discussion.

**Hypothesis:** weak bridges between strongly coupled local blocks create slow collective modes. If those bridges define a relatively stable *address scaffold*, local material changes can store content without globally rewriting that scaffold. An exact local forward/adjoint sensitivity gives a supervised way to write the material. We then test whether changing the bridges damages the result.

**Status: exploratory CPU toy with one fixed teacher and one write order; not a demonstrated new AI architecture, a biological lipizone model, or a learned PAC code.** The results below are from a local run of the supplied deterministic code and can be regenerated. The gates were not preregistered before running.

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
