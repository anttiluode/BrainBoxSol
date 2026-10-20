# Bridge-robustness implementation review

Base: `c706ab541109541303211efaa5ad31e07733184e`. Reviewed implementation/fixes: `7cb745c7d0b197742616401a5d1838d13f19854e`.

A read-only independent code reviewer found two Important issues: coherent-interface drift is exactly reversed by the repair and makes the combined comparison degenerate; and receipt audit accepted an arbitrary nonempty lock label without resolving the recorded commit.

Both were fixed before full seeds. The primary is now independent per-edge drift, with the original protocol and development outcome preserved. Relative effects below a fixed 1e-8 control-error floor are unresolved and cannot pass. Full receipt audits resolve the exact Git commit, hash its source blobs and compare those to the recorded/current implementation.

Two minor issues were also fixed: deterministic update/bridge-write counts are now audited, and finite-difference tests include nonunit writable conductance factors. Intermediate training losses are stored but explicitly not independently recomputed without retraining.

Regression evidence: all three added negative tests failed on the original implementation (forged update count, forged source commit, numeric-zero relative effect). The repaired implementation passed all **63** repository tests in [GitHub Actions run 39471990982](https://github.com/anttiluode/BrainBoxSol/actions/runs/39471990982). The reviewer then reported no remaining Critical or Important findings. No tests or held-out settings were changed to obtain a favorable scientific result.

## Independent training-audit review

A separate read-only reviewer inspected `reproduce_bridge_training.py`, its regression tests, the portable endpoint auditor, method updates and CI. The reviewer also independently ran the complete training-audit CLI against the clean source lock: **20 writers, 900 conductances and 960 intermediate losses reproduced**, with the original receipt bytes unchanged. Maximum absolute field difference was 2.927125208884718e-11; maximum loss-history difference 1.7347776860854314e-15. No Critical or Important issue was found.

The reviewer confirmed that transparent instrumentation measures actual read/node/update solves while preserving numerical calls, and that combined forward/adjoint counting is disclosed. This independently executes the same locked implementation; it is not an independent optimizer implementation. No new held-out scientific claim is introduced.

Two minor findings were addressed: historical wording calling 66 tests the final branch count was corrected; runtime environment collection now tolerates the absence of a NumPy-version-specific build-configuration attribute on supported NumPy 1.x versions. A regression test first failed for the missing compatibility collector, then passed with the guarded fallback. The complete suite contains 74 tests. The original training audit is preserved alongside a new audit made after this logging-only change; each records its producing auditor hash. Neither change modifies scientific source, comparison tolerances or results.
