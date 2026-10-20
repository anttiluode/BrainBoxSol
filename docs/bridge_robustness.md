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

## Locked fresh-device outcome

**Both predeclared benefits failed.** Exact source/protocol lock: `7cb745c7d0b197742616401a5d1838d13f19854e`. Full run: [39472165848](https://github.com/anttiluode/BrainBoxSol/actions/runs/39472165848). No settings changed after this lock. Ten untouched devices (7001–7010), twelve paired draws per family/strength.

Primary independent bridge drift, log-noise strength 0.6; mean absolute target-response error normalized by the device's initial response (lower is better):

| Arm | Mean target error | Mean change from its own trained response |
|---|---:|---:|
| Ordinary writing | 0.02617034 | 0.02617034 |
| Noise-aware writing | 0.02729738 | 0.02623119 |
| Interface repair | 0.00110702 | 0.00110702 |
| Noise-aware + repair | 0.00369595 | 0.00107218 |

Noise-aware writing is **4.31% worse** than ordinary writing and improves only 1/10 device means. Combined writing/repair is **3.34× the target error** of repair alone and improves 0/10. Repair alone reduces ordinary error by **95.77%**. A frozen untrained material scores 0.02713311.

Ordinary clean error is 3.814e-14, noise-aware clean error 0.00382452. The 0.00382452 clean penalty exceeds the allowed 0.002. Device-paired bootstrap 95% intervals for error reduction are negative: noise-aware vs ordinary [-0.00169868, -0.00058963]; combined vs repair [-0.00355126, -0.00164765]. These intervals are descriptive and resample devices, not ports or drift trials.

The combined arm changes slightly less from its own trained output than repair alone, yet is less correct because its clean target fit is biased. **Response stability and task accuracy are different measurements.** This is why both were recorded.

Each writer used 2352 forward and 2352 adjoint solves plus 48 update solves per device, identically across arms. Cost matching does not make this competitive with a direct digital target table.

The original numerical audit recomputed all **3840** paired arm evaluations, regenerated RNG inputs, verified source hashes and checked endpoint losses and deterministic training costs. All **63 repository tests** passed at the scientific source lock. That original audit did not retrain. The separate training reproduction below now covers the complete optimization trajectories. Both the fresh receipt and all development versions remain available.

[Full receipt](../results/bridge_robustness/full.json) · [Self-contained visual report](../results/bridge_robustness/report.html) · [Implementation review](bridge_robustness_review.md).

Interpretation: this particular noise-aware writer did not transfer a useful robustness benefit to fresh physical drift, despite the reasonable analogy to NCM conditioning corruption. It does not rule out all noise-aware methods or larger training banks. Do not retune these full seeds and re-label the result held out.

## Post-lock audit portability

The original scientific source and full JSON remain unchanged. A PR runner regenerated one initial bridge as 0.0306364121715947 instead of the recorded 0.030636412171594705, a two-ULP difference (6.94e-18), so the original bit-exact bridge check rejected an otherwise matching receipt. The producing-run and another runner's strict audits passed. The earlier description called this one ULP; this corrects the wording without changing any result.

`audit_bridge_receipt.py` is a separate post-lock audit utility. It allows at most eight ULPs of initial-bridge regeneration difference, reports every adjustment and its maximum size, normalizes only those entries in a deep audit copy, then calls all original source-provenance, physics, RNG, endpoint, cost and summary checks. It never changes the producing source or saved receipt. Larger material changes are rejected.

Tests first reproduced the strict one-ULP failure, then verify that the portable audit reports it without mutating input and rejects a 1e-10 material change. This is numerical portability handling, not a change to training or scientific verdicts. The original full run had 63 tests; the earlier portability branch added three tests, for 66. The training auditor adds eight tests, bringing the suite to 74.

## Independent locked-training reproduction

The [separate training audit](../results/bridge_robustness/training-audit-portable.json) reruns the exact clean source commit `7cb745c7d0b197742616401a5d1838d13f19854e` in a fresh Python interpreter. It does not initialize from saved trained materials. Both writers start from each device's regenerated initial field, teacher and training bank, using the unchanged protocol and seeds 7001–7010. The original receipt's SHA-256 remains `ae4e7c1288e5b73a20cb0af40d15fa6e9473025aad6931c65e4c22a8916821e3`.

The [initial training audit](../results/bridge_robustness/training-audit.json) is also retained. Its producing auditor is preserved in commit `ad3cc9bdb33ae91079ea91e62893cf6519563f53`. A subsequent environment-logging compatibility fix was verified with a new output file, never by overwriting the initial audit. The training source, comparisons and thresholds did not change.

All **20 writers**, **900 final conductances**, and **960 intermediate losses** reproduced within fixed numerical tolerances. Maximum absolute field difference was **2.9272e-11**; maximum loss-history difference **1.7348e-15**; maximum endpoint-loss difference **1.0634e-15**. Conductance tolerance is `atol=1e-11, rtol=1e-9`; losses use `atol=1e-15, rtol=1e-8`. These are numerical reproducibility thresholds, not scientific benefit thresholds. Source and protocol hashes, flags and seeds must match exactly; solve counts must match exactly.

Transparent call counters around the locked writer observe **2352 reads, 4704 node-system linear solves and 48 update solves per writer/device**. The locked `read` executes one forward and one adjoint solve each time; the counter records their aggregate rather than inferring costs solely from the saved formula. Both ordinary and noisy writers use the same actual budget. All four recomputed primary mean target errors differ from the original by at most **1.902e-14**. Both failed benefit claims remain failed.

The original and rerun receipts each also pass the portable endpoint audit. On this runner, 16 original bridge entries (counting both writer copies) differ from regenerated initials by at most two ULPs, or 1.388e-17. Only the audit copy is normalized; all original materials and results remain intact. Python/NumPy/BLAS environment, source hashes, tolerances, rerun fields, full loss histories and observed counts are recorded in the separate audit. This is independent execution of the locked implementation, not a separately implemented optimizer or new held-out experiment.

Reproduce from a full Git clone, with a new output path each time:

```bash
git worktree add --detach ../brainbox-locked 7cb745c7d0b197742616401a5d1838d13f19854e
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reproduce_bridge_training.py \
  --source ../brainbox-locked \
  --receipt results/bridge_robustness/full.json \
  --out /tmp/bridge-training-audit-new.json
```

## Follow-up decision

No additional scientific gate is introduced in this audit. NCM-inspired corruption training has now failed its two locked benefit tests; changing noise settings on these ten seeds would be development, not fresh confirmation. A larger training bank is not justified merely by this negative result.

HybridAgent-inspired quantization is a useful question only after specifying the communication boundary. Here raw charge already gives 100% binary-task accuracy and the matched-waveform peak gives 96.6406%. Thus the existing task does not establish that a rich waveform is necessary. One nearest word is also not one bit: an index into a vocabulary of size V needs approximately log2(V) bits. A receiver-aligned projection followed by a trained one-bit threshold could reproduce an existing binary decision by construction; that alone would not discover a stronger memory mechanism.

A future quantization gate would need a fresh unresolved task, training-only codebook/calibration, transmitted index bit budgets, and explicit accounting for amplitude/charge, scale factors, timing and any other continuous side channel. No new paper performance claims are relied upon here. For now the useful result is the retained falsification and complete training reproduction, rather than another tuned model.
