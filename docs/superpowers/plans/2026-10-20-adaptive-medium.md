# Adaptive Material Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. Native execution was approved by the user at 12:02 Europe/Helsinki; completed steps are checked below.

**Goal:** Demonstrate whether supervised changes to a small conductance grid cause better later cue-to-motor responses after fast activity is reset.

**Architecture:** Separate transient node activity from persistent edge conductances. Train the material with the exact discrete adjoint under a bounded fixed budget; evaluate through fixed motor regions with erased, shuffled, frozen, traffic-only and conventional controls. Produce raw receipts and a data-backed HTML replay.

**Tech Stack:** Python 3.10+, NumPy, unittest; standard-library JSON/HTML generation and GitHub Actions. No new runtime dependencies.

**Spec:** [Approved adaptive-medium design](../specs/2026-10-20-adaptive-medium-design.md), approved by the user on 2026-10-20 at 11:50 Europe/Helsinki.

## Global Constraints

- CPU-scale NumPy implementation; no modification of BrainBoxClaude or existing G0-G3 behaviour.
- A 9 by 9 open-boundary grid: 81 activity nodes, with horizontal, vertical and both diagonal candidate edges between neighbours; no periodic wrap.
- Initial numerical settings: leak 0.02, time step 0.1, initial mean conductance 0.08, minimum 0.005, maximum 0.4.
- Require `dt * (2*d_max + leak) <= 0.8`; reject invalid settings instead of silently changing the protocol.
- Material bounds within 1e-12 and total-budget error at most 1e-10 times the budget.
- Episode: 160 steps, sensory source for the first 24; integrate motor states 113 through 160, inclusive.
- Fixed score: `(E0-E1)/(E0+E1+1e-9)`; exact ties are incorrect; total motor response <=1e-8 is a failure.
- Training loss: `0.5*(r - 0.8*z)**2`; initial learning rate 0.05 and 600 minibatch updates, one example of each cue per minibatch.
- Development seeds: 11, 23, 41. Held-out seeds: 101 through 110; both mappings for each seed; 100 fresh evaluation examples per cue and 20 shuffled-material permutations.
- Only development data may inform protocol changes. Commit the complete protocol and evaluation code before the held-out run.
- Evaluation clears activity before each episode, freezes material and receives no teacher label in the acting path.
- Gate: mean learned accuracy >=0.90, minimum seed accuracy >=0.80, gain over frozen >=0.20, erasure removes >=half that gain, gain over shuffled >=0.10, plus all numerical and episode-validity requirements.
- Fixed candidate edges and supervised reverse-mode differentiation must be named accurately; no claim of spontaneous neurites, autonomous credit, a complete brain or better digital AI.

## Review Focus

1. Invalid or nonfinite parameters and material, infeasible budgets, or an unstable time step: raise a useful ValueError before simulation (Task 1).
2. A tiny or zero motor response and an exact tie: count an incorrect episode instead of manufacturing confident accuracy (Tasks 2 and 4).
3. Cross-episode state, teacher labels or evaluation draws leaking into later training: isolated arrays, acting interfaces without labels, independent deterministic streams (Tasks 2-4).
4. A corrupted material checkpoint or held-out execution from uncommitted code: reject it; record exact provenance for valid runs (Tasks 3-4).
5. Missing or malformed replay data and a browser without scripting: readable fallback, no fabricated frames or silently blank output (Task 5).

## Files and boundaries

| File | Responsibility |
|---|---|
| `adaptive_medium.py` | Grid, operator, stable rollout, budget projection, transient adjoint |
| `adaptive_medium_task.py` | Immutable task constants, cue generation, drive timing, fixed motor readout |
| `adaptive_medium_gate.py` | Training, interventions, conventional controls, evaluation, CLI and JSON receipts |
| `adaptive_medium_replay.py` | Convert recorded episodes and measurements to self-contained HTML |
| `test_adaptive_medium.py` | Numerical, gradient, task, reset and serialization tests |
| `test_adaptive_medium_gate.py` | Controls, aggregation, protocol guard, CLI and replay tests |
| `protocols/adaptive_medium.json` | All frozen experiment constants, seeds, thresholds and protocol version |
| `docs/adaptive_medium.md` | Development changes, locked protocol, measured results and limitations |
| `README.md`, `.github/workflows/tests.yml` | Run commands, status and small CI smoke run |

Existing modules remain untouched. Use ordinary dataclasses and NumPy arrays;
no framework, package migration, application server or new dependency is needed.

### Task 1: A stable spatial medium with a fixed material budget

**Files:** Create `adaptive_medium.py`; create the numerical section of `test_adaptive_medium.py`.

**Interfaces:**
- `Grid(size: int, edges: ndarray[E,2])`, property `nodes: int`.
- `make_grid(size: int = 9) -> Grid`; each undirected neighbour pair occurs once, yielding 272 edges at size 9.
- `project_budget(values: ndarray[E], budget: float, lower: float, upper: float) -> ndarray[E]`.
- `initial_material(grid: Grid, seed: int, *, mean: float = .08, lower: float = .005, upper: float = .4) -> ndarray[E]`; seeded uniform perturbation in [0.9,1.1] times mean, then projection.
- `operator(grid: Grid, g: ndarray[E], *, leak: float) -> ndarray[N,N]` implements `B diag(g) B.T + leak*I`.
- `rollout(grid: Grid, g: ndarray[E], drive: ndarray[T,N], *, dt: float, leak: float) -> ndarray[T+1,N]`; first state is newly allocated zeros. No caller-supplied initial state or persistent activity cache.

- [x] **Step 1: Write the failing numerical tests.** Assert the following independent properties, including the stated invalid inputs:

```python
assert make_grid(9).nodes == 81
assert len(make_grid(9).edges) == 272
np.testing.assert_allclose(A, A.T, atol=1e-14)
assert np.linalg.eigvalsh(A).min() > 0  # positive leak
np.testing.assert_allclose(no_leak_states.sum(axis=1), initial_mass, atol=1e-12)
assert np.linalg.norm(leaky_states[-1]) < np.linalg.norm(leaky_states[1])
assert abs(projected.sum() - budget) <= 1e-10 * budget
```

Use a one-step source followed by no drive to establish initial_mass; do not expect an all-zero initial state to decay. A corner impulse may reach only its three graph neighbours in the next state, proving no wrap. Assert ValueError for size <2, incorrect shape, NaN/Inf material or drive, negative leak, dt<=0, the violated stability bound and budgets outside `[E*lower,E*upper]`. Assert projection handles boundary budgets and does not mutate its input.
- [x] **Step 2: Run `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m unittest -v test_adaptive_medium`.** Expect an import failure before implementation.
- [x] **Step 3: Implement the declared interfaces.** Use float64. A dense 81-node operator is acceptable; projection uses clipped scalar-threshold bisection. Validate finite values before constructing the operator or updating state.
- [x] **Step 4: Rerun the same tests.** Expect all new numerical tests to pass with the stated tolerances.
- [x] **Step 5: Commit only this tested numerical core and its tests.** Message: `feat: add stable budgeted spatial conductance medium`.

### Task 2: Sensory episodes, motor readout and the exact transient writer

**Files:** Create `adaptive_medium_task.py`; extend `adaptive_medium.py` and `test_adaptive_medium.py`.

**Consumes:** Task 1 Grid, operator, projection and rollout.
**Produces:**
- Frozen `TaskConfig` dataclass containing every source, timing, readout and noise constant from the spec; `asdict` is JSON-compatible. Default size is 9.
- `sample_cue(cue: int, rng: Generator, cfg: TaskConfig, *, evaluation: bool) -> ndarray[N]`; no mapping or target argument.
- `make_drive(source: ndarray[N], cfg: TaskConfig) -> ndarray[T,N]`.
- `read_motor(states: ndarray[T+1,N], cfg: TaskConfig) -> Episode`; Episode has `score: float`, `energies: ndarray[2]`, `action: int | None`, `valid: bool`, `failure: str | None`.
- `loss_gradient(grid: Grid, g: ndarray[E], drive: ndarray[T,N], cfg: TaskConfig, target_sign: int) -> tuple[float, Episode, ndarray[E]]`.
- `motor_response_map(grid: Grid, g: ndarray[E], cfg: TaskConfig) -> ndarray[2,N]`; exact fixed-material solver shortcut, not a learned decoder.

- [x] **Step 1: Write the failing task and gradient tests.** Verify each cue stays inside its specified 3 by 3 patch, both sources are nonnegative, amplitudes are in [0.8,1.2], seeded draws reproduce exactly, and training/evaluation jitter and noise differ as specified. Check exactly 24 nonzero drive states and motor integration indices 113..160 with dt weighting.

```python
np.testing.assert_array_equal(states[0], np.zeros(81))
np.testing.assert_array_equal(g, before_g)
np.testing.assert_allclose(analytic_edges, central_difference_edges,
                           rtol=2e-4, atol=1e-7)
np.testing.assert_allclose(response_map @ source, episode.energies,
                           rtol=1e-10, atol=1e-12)
assert zero_response.action is None and not zero_response.valid
assert balanced_response.action is None  # exact nonzero tie
```

Use central-difference step 1e-6 on several non-boundary conductances, both target signs and two distinct materials. Repeat A, then B, then A with unchanged material; both A rollouts must be identical. Perturb one input array after a run and ensure stored earlier results do not change.
For refinement, the original right-endpoint motor quadrature covers (11.2,16.0]; at dt=0.05 integrate states 225..320, with forcing for the first 48 steps. Do not simply double the original starting index.
- [x] **Step 2: Run the affected unittest module.** Expect missing-interface failures.
- [x] **Step 3: Implement the task and adjoint interfaces.** Use the spec's full ratio derivative and time-indexed adjoint, including q_T. Do not use clipping to hide nonfinite gradients. For the response shortcut, propagate the two fixed motor functionals backwards, accumulate their sensitivities to the first 24 source steps and compare to full rollouts. Its output has no class labels or fitted parameters.
- [x] **Step 4: Run numerical, cue and gradient tests.** Require every finite-difference and exact-map assertion to pass. Assert a material reset reproduces initial scores; no target may be passed to `rollout`, `sample_cue`, `read_motor` or `motor_response_map`.
- [x] **Step 5: Commit.** Message: `feat: add sensory task and exact transient material gradient`.

### Task 3: Supervised training, traffic reinforcement and matched-input controls

**Files:** Create the training/control section of `adaptive_medium_gate.py`; extend `test_adaptive_medium.py`; create `test_adaptive_medium_gate.py`.

**Consumes:** Tasks 1-2 interfaces.
**Produces:**
- `MaterialRun(initial: ndarray[E], final: ndarray[E], curve: list[dict], costs: dict)`.
- `train_material(grid: Grid, seed: int, mapping: tuple[int,int], cfg: TaskConfig, *, updates: int = 600, rate: float = .05, mode: str = 'adjoint') -> MaterialRun`; mode is `adjoint` or `traffic`.
- `make_evaluation_sources(seed: int, cfg: TaskConfig, per_cue: int = 100) -> tuple[ndarray[2*per_cue,N], ndarray[2*per_cue]]`; second array is evaluator-only cue identity.
- `evaluate_material(grid: Grid, g: ndarray[E], sources: ndarray[K,N], cfg: TaskConfig) -> list[Episode]`; no labels/mapping accepted.
- `train_linear(seed: int, mapping: tuple[int,int], cfg: TaskConfig, *, updates: int = 600, rate: float = .05) -> tuple[ndarray[2,2], ndarray[2], dict]`; same training examples as the medium, softmax cross-entropy, zero initial weights and biases.
- `save_material(path: Path, g: ndarray[E], cfg: TaskConfig) -> None`; `load_material(path: Path, grid: Grid, cfg: TaskConfig) -> ndarray[E]` uses JSON, validates metadata/shape, bounds, budget and finiteness.

Mapping tuples are `(0,1)` and `(1,0)`: entry `mapping[cue]` is the target motor, and the loss sign is `1-2*mapping[cue]`.

- [x] **Step 1: Write failing trainer and control tests.** With a very small update count, require deterministic equality on rerun, unchanged initial g, bounded/fixed-budget final g, finite curves and changed supervised material. Do not assert the full research gate succeeds. Both modes and the classifier must see byte-identical recorded cue draws; evaluation uses a separate SeedSequence child stream and never changes learned parameters.

```python
np.testing.assert_array_equal(reloaded_g, trained_g)
np.testing.assert_array_equal(reloaded_episode.energies, original_episode.energies)
np.testing.assert_array_equal(erased_g, run.initial)
np.testing.assert_array_equal(np.sort(shuffled_g), np.sort(run.final))
```

Reject checkpoint wrong grid size, malformed JSON, NaN and incorrect budget. Verify exact ties and invalid episodes are scored wrong; reversing evaluator labels cannot change the returned motor responses. Count two forward and two adjoint rollouts per supervised minibatch, two forward rollouts for traffic, and zero adjoints for traffic.
- [x] **Step 2: Run `python -m unittest -v test_adaptive_medium_gate`.** Expect missing-interface failures.
- [x] **Step 3: Implement training and controls.** Average the two cue gradients before projection. Traffic uses positive average squared forward edge drops with the same initial rate and budget. Derive stream identifiers from seed and purpose; do not consume the evaluation stream while training. Measure actual solver work, checkpoint bytes, full-trajectory storage and elapsed time. The response map can accelerate evaluation only after Task 2 equivalence tests pass; report its actual build/query cost.
- [x] **Step 4: Rerun both adaptive test modules.** Require serialization, causal intervention construction and leakage tests to pass. The two-entry lookup gets true cue identity, is explicitly an oracle and has no material training cost.
- [x] **Step 5: Commit.** Message: `feat: train adaptive material with causal and digital controls`.

### Task 4: Reproducible gate, development evidence and protocol lock

**Files:** Complete `adaptive_medium_gate.py`; create `protocols/adaptive_medium.json` and `docs/adaptive_medium.md`; extend `test_adaptive_medium_gate.py`.

**Consumes:** Task 3 training, material evaluation and control interfaces.
**Produces:**
- `evaluate_seed(seed: int, cfg: TaskConfig, *, updates: int, per_cue: int, shuffle_count: int) -> dict`; both mappings, all seven arms and per-mapping raw results.
- `summarize(seed_results: list[dict], *, numerical_checks: dict) -> dict`; averages mappings and permutations inside each seed, then seeds equally. Produces each conjunct, overall pass/fail and all failed reasons.
- `main(argv: list[str] | None = None) -> int`; CLI modes `development`, `smoke`, `full`, with `--out`, `--protocol` and full-only `--lock-commit`. Return 0 for a valid completed run even if its scientific gate fails; return 2 for invalid inputs or provenance. Numerical execution failures get an explicit failed receipt when possible and a nonzero status.

- [x] **Step 1: Write failing gate/CLI tests.** Use synthetic metric dictionaries to prove all five verdict conjuncts and each failure condition independently. Test unequal sample counts cannot give one seed extra weight. Low-energy examples remain in denominators, failures never disappear and exactly 20 deterministic permutations are used in full mode. Smoke can never label itself a full held-out pass.
- [x] **Step 2: Run gate tests and observe the expected missing functionality.** Include full-mode rejection for missing lock, dirty tracked files, wrong commit, code/JSON absent from that commit and any development seed entering the held-out list. Reject NaN receipt metrics; JSON serialization uses `allow_nan=False`.
- [x] **Step 3: Implement gate orchestration and provenance guards.** Full mode requires `--lock-commit` equal to current HEAD, a clean tracked tree and a tracked `locked: true` protocol at that commit. Include protocol SHA256 and code commit in receipts. A lock file never embeds its own commit hash. Read labels only in evaluation aggregation.
- [x] **Step 4: Run development only on seeds 11, 23, 41, both mappings.** Start with spec defaults and record curves, gradient checks, energy floors, all controls and wall time. Run dt/2 with doubled step counts and the same physical forcing/observation windows on both initial and trained material. Require maximum score difference <0.02 before locking. If defaults fail, diagnose using development data; document every changed constant. Do not run seeds 101..110 or change verdict thresholds.
- [x] **Step 5: Run adaptive tests and review the actual development report.** A learning failure is a reported research outcome, not grounds to weaken the gate. Record every seed and both mappings even when an average looks favourable.
- [x] **Step 6: Commit development implementation and evidence.** Message: `experiment: document adaptive-medium development protocol`. The later lock commit is deferred to Task 6 so replay and CI code also exist before full evaluation.

### Task 5: A visible, self-contained replay of actual episodes

**Files:** Create `adaptive_medium_replay.py`; extend `test_adaptive_medium_gate.py`; add a replay invocation to the gate CLI.

**Consumes:** Grid, material and recorded trajectories from Tasks 1-4; raw motor responses and receipt metadata.
**Produces:** `write_replay(path: Path, *, grid: Grid, receipt: dict, episodes: list[dict]) -> None`; each episode contains label text, initial/final material identity, cue, source, actual sampled states, time samples and raw motor energies. Gate flag `--replay PATH` generates before/after/frozen/erased episodes for development seed 11; full runs select seed 101 without tuning to the best seed.

- [x] **Step 1: Write failing replay tests.** Assert all sampled state values round-trip from embedded JSON, all labels/metadata are escaped, no external script/font/CDN is referenced, and a static results table plus noscript explanation exists. A literal `</script>` in metadata cannot break embedding. Missing or inconsistent episode arrays raise ValueError; do not substitute fake activity.
- [x] **Step 2: Run the replay tests.** Expect a missing module/interface failure.
- [x] **Step 3: Implement the HTML renderer.** Use canvas for edge widths/material and node activity with play/pause, cue/arm selection and time scrubbing. Show physical time, seed, mapping, fixed material budget, motor energies and the fact this is a replay. Bundle its dataset. Thresholded edges disclose the threshold; no training occurs inside the replay.
- [x] **Step 4: Generate a development replay and visually inspect it.** Use available browser tooling if available; otherwise render deterministic snapshots of the same data for visual QA and record that interactive browser behaviour is unverified. Check narrow/wide layouts, controls, label readability and correspondence with the JSON. Do not add a dependency solely to launch a browser.
- [x] **Step 5: Rerun replay tests and commit.** Message: `viz: add data-backed adaptive-medium replay`.

### Task 6: CI, lock, held-out run and a measured verdict

**Files:** Modify `.github/workflows/tests.yml` and `README.md`; finalize `protocols/adaptive_medium.json` and `docs/adaptive_medium.md`; add compact raw receipts and HTML replay under `results/adaptive_medium/`.

**Consumes:** All preceding tasks and numerical checks. No new acting or learning interfaces.

- [x] **Step 1: Add the adaptive unittest modules and a smoke command to CI.** Smoke uses development seed 11, both mappings, 5 updates, 3 examples per cue and 2 permutations; its receipts explicitly say non-held-out and cannot satisfy the full gate. Preserve the existing G0-G3 steps. Upload new nested JSON/HTML artifacts as well as existing receipts.
- [x] **Step 2: Run `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -v` and the smoke command.** Require all tests to pass and a complete finite smoke receipt; do not require the research gate to pass at 5 updates.
- [x] **Step 3: Commit the complete protocol lock before any held-out training.** Include all implementation/evaluation/replay code, CI, exact constants and `locked: true`. Message: `experiment: lock adaptive-medium held-out protocol`. Record the resulting SHA for the next command.
- [x] **Step 4: Run `OPENBLAS_NUM_THREADS=1 python adaptive_medium_gate.py --mode full --protocol protocols/adaptive_medium.json --lock-commit <lock-SHA> --out results/adaptive_medium/full.json --replay results/adaptive_medium/replay.html`.** Use seeds 101..110, both mappings, 100 examples per cue and 20 permutations. Keep progress messages at least once per seed/mapping. Do not tune or rerun selected seeds to improve the result. Resume only by reusing completed receipts with matching lock SHA and protocol digest, preserving failures.
- [x] **Step 5: Write the measured report and README status.** Include every arm/seed, mean/std/range, raw energy failures, all verdict conjuncts, development changes, actual compute/storage and the limitations from the spec. Report failure without spin if any conjunct fails. Distinguish green software checks from the scientific gate outcome.
- [x] **Step 6: Rerun the full unittest suite and inspect the replay and receipt.** A JSON round-trip must preserve metrics; stored/reloaded material must reproduce the fixed episode. Include the implementation's selected review method, address concrete review findings, then publish the result on the BrainBoxSol branch as a reviewable PR update. Do not merge earlier PRs or touch another repository.
- [x] **Step 7: Commit the receipts, replay and final report.** Message: `experiment: report locked adaptive-material memory result`. Link the PR, run command, measured outcome and visual replay in the final response.

## Plan self-review and handoff

Coverage: Tasks 1-2 cover the operator, bounds, source/readout and gradient;
Task 3 covers writes, all controls and persistent serialization; Task 4 covers
replication, validity, independent streams, numerical refinement and provenance;
Task 5 covers the visible replay; Task 6 covers CI, the pre-evaluation lock,
measurements and honest publication. All five Review Focus entries have owning
tests. The saved-material reset and the full-rollout/response-map equivalence
checks prevent two especially plausible false positives.

Execution method approved: **Native**. Tasks 1-6 are complete; development and held-out measurements are documented in `docs/adaptive_medium.md`. These
six tasks share a small set of tightly coupled numerical interfaces; one
implementer can preserve the definitions and check the whole branch with a
fresh reviewer at the end. The alternative is subagent-driven implementation
with a fresh implementer/reviewer for each task. Native execution and one independent whole-branch review were completed. The review fixes and remaining limitations are recorded in `docs/adaptive_medium_review.md`.
