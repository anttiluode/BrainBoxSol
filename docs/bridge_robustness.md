# Bridge robustness: noise-aware writing × interface repair

Authorized October 20 extension of BrainBoxSol. This compares four arms on the existing `PhaseMedium`: ordinary supervised writing, writing under bridge perturbations, interface-total repair, and their combination. Claude's repository is not modified.

The NCM analogy is training under imperfect conditioning. This is ordinary noise-aware optimization, not evidence that a learner must discover an invariant subspace.

## Frozen design

All arms use the same simultaneous, damped Gauss–Newton writer on the same six scalar teacher targets and the same positive interior conductances. It uses the existing exact forward/adjoint edge derivatives; global least-squares computation, target access and normalization are explicit. This does not reproduce the earlier sequential OGD writer.

The standard writer evaluates eight identical clean worlds; the noise-aware writer evaluates eight fixed mixed coherent/independent bridge-noise worlds. Both perform the same 48 updates and actual response/Jacobian computations. Bridge conductances themselves never change during writing. Evaluation compares the two resulting materials with/without interface-total repair on exactly paired fresh drifts.

Repair stores one pre-drift conductance total per interface and rescales the drifted bridges. It has no targets or replay. Test families isolate independent bridge drift, coherent interface drift, alternating-interface drift, and interior drift as an out-of-scope diagnostic, each at log-noise strengths 0.2 and 0.6. Clean-fit costs are reported alongside drift robustness.

Primary claims are specified in [the protocol](../protocols/bridge_robustness.json): at coherent drift 0.6, does noise-aware writing improve ordinary writing, and does the combination improve repair alone? Each claim requires at least 10% mean error reduction, benefit in at least 80% of fresh devices, and clean normalized target-error cost at most 0.002. These thresholds can fail.

Device means are replication units. Paired differences and bootstrap intervals are descriptive; ports and repeated drift draws are not independent devices. The exact protocol/source commit must be frozen before running seeds 7001–7010. Development seeds are separate. No full-seed retuning.

## Verification and receipts

Tests check the corrupted objective's chain-rule derivative against central differences, bridge immutability, zero-noise equivalence, matched computation counts, repair constraints, seed separation, source locks and receipt recomputation/tamper rejection. The existing full repository test suite must also pass.

The runner provides smoke/development modes and a full mode requiring an exact clean lock commit. Full receipts retain conductances, targets, training noise banks, evaluation noise factors and paired errors so the numerical summary can be recomputed. A small HTML report exposes all arms and limits.

## Scope

This is a tiny known-port regression task with a supplied teacher. Successful robustness does not establish autonomous learning, new capacity, biological plasticity or an advantage over digital indexed storage. Noise-aware writing may sacrifice clean fit; interface repair cannot undo interior damage. A negative combined result is retained.
