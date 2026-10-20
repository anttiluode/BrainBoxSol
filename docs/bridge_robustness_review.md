# Bridge-robustness implementation review

Base: `c706ab541109541303211efaa5ad31e07733184e`. Reviewed implementation/fixes: `7cb745c7d0b197742616401a5d1838d13f19854e`.

A read-only independent code reviewer found two Important issues: coherent-interface drift is exactly reversed by the repair and makes the combined comparison degenerate; and receipt audit accepted an arbitrary nonempty lock label without resolving the recorded commit.

Both were fixed before full seeds. The primary is now independent per-edge drift, with the original protocol and development outcome preserved. Relative effects below a fixed 1e-8 control-error floor are unresolved and cannot pass. Full receipt audits resolve the exact Git commit, hash its source blobs and compare those to the recorded/current implementation.

Two minor issues were also fixed: deterministic update/bridge-write counts are now audited, and finite-difference tests include nonunit writable conductance factors. Intermediate training losses are stored but explicitly not independently recomputed without retraining.

Regression evidence: all three added negative tests failed on the original implementation (forged update count, forged source commit, numeric-zero relative effect). The repaired implementation passed all **63** repository tests in [GitHub Actions run 39471990982](https://github.com/anttiluode/BrainBoxSol/actions/runs/39471990982). The reviewer then reported no remaining Critical or Important findings. No tests or held-out settings were changed to obtain a favorable scientific result.
