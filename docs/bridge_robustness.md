# Bridge robustness: noise-aware writing × interface repair

Authorized October 20 extension of BrainBoxSol. This compares four arms on the existing `PhaseMedium`: ordinary supervised writing, writing under bridge perturbations, interface-total repair, and their combination. Claude's repository is not modified.

The NCM analogy is training under imperfect conditioning. This is ordinary noise-aware optimization, not evidence that a learner must discover an invariant subspace.

## Frozen design

All arms use the same simultaneous, damped Gauss–Newton writer on the same six scalar teacher targets and the same positive interior conductances. It uses the existing exact forward/adjoint edge derivatives; global least-squares computation, target access and normalization are explicit. This does not reproduce the earlier sequential OGD writer.

The standard writer evaluates eight identical clean worlds; the noise-aware writer evaluates eight fixed mixed coherent/independent bridge-noise worlds. Both perform the same 48 updates and actual response/Jacobian computations. Bridge conductances themselves never change during writing. Evaluation compares the two resulting materials with/without interface-total repair on exactly paired fresh drifts.

Repair stores one pre-drift conductance total per interface and rescales the drifted bridges. It has no targets or replay. Test families isolate independent bridge drift, coherent interface drift, alternating-interface drift, and interior drift as an out-of-scope diagnostic, each at log-noise strengths 0.2 and 0.6. Clean-fit costs are reported alongside drift robustness.

Primary claims are specified in [the protocol](../protocols/bridge_robustness.json): at independent drift 0.6, does noise-aware writing improve ordinary writing, and does the combination improve repair alone? Each claim requires at least 10% mean error reduction, benefit in at least 80% of fresh devices, and clean normalized target-error cost at most 0.002. These thresholds can fail.

Device means are replication units. Paired differences and bootstrap intervals are descriptive; ports and repeated drift draws are not independent devices. The exact protocol/source commit must be frozen before running seeds 7001–7010. Development seeds are separate. No full-seed retuning.

## Verification and receipts

Tests check the corrupted objective's chain-rule derivative against central differences, bridge immutability, zero-noise equivalence, matched computation counts, repair constraints, seed separation, source locks and receipt recomputation/tamper rejection. The existing full repository test suite must also pass.

The runner provides smoke/development modes and a full mode requiring an exact clean lock commit. Full receipts retain conductances, targets, training noise banks, evaluation noise factors and paired errors so the numerical summary can be recomputed. A small HTML report exposes all arms and limits.

## Scope

This is a tiny known-port regression task with a supplied teacher. Successful robustness does not establish autonomous learning, new capacity, biological plasticity or an advantage over digital indexed storage. Noise-aware writing may sacrifice clean fit; interface repair cannot undo interior damage. A negative combined result is retained.

## Development-informed amendment before fresh seeds

V1 chose coherent interface drift as the primary condition. On development seeds, repair returned ordinary responses to numerical precision: each interface's common multiplier cancels exactly. Thus that comparison tested clean fit rather than a nontrivial robustness gain. Independent per-edge drift at 0.6 is now the primary condition, with coherent drift retained as a mechanism check. The original protocol is archived at `protocols/archive/bridge_robustness_v1.json`; its exact source is `0f1033a525e361cbecdf942b73bfeed134edb031` and its receipt is `results/bridge_robustness/development-v1.json`.

The v1 development result was unfavorable to noise training: ordinary clean error was about 3e-14, noise-aware error 0.00397; independent-drift error at 0.6 was 0.02309 ordinary, 0.02474 noise-aware, 0.001176 repair, and 0.003979 combined. No learning hyperparameter was changed in response. Both proposed benefits may fail on fresh seeds.

A relative change with control error below 1e-8 is now marked unresolved, not divided by a numerical floor to manufacture huge percentages. The source-lock audit resolves the recorded commit and checks its actual source blobs. Training endpoint objectives, immutable bridges and deterministic solve counts are audited; intermediate optimization losses are stored but not independently recomputed without retraining.
