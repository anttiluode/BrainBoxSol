# History waveform implementation review

A read-only independent reviewer checked the new experiment before protocol locking and did not evaluate fresh seeds. No critical or important issues were found. The reviewer confirmed physical boundary isolation, paired controls, training-only threshold calibration, matching invariants, numerical-refinement scope, memory accounting, and exact committed-source protection. All eight then-existing new tests passed independently.

The reviewer identified a minor output-path hazard: full and smoke modes defaulted to development.json. A regression test reproduced an overwritten development sentinel. Mode-specific output defaults fixed it. The complete suite now has 49 tests.

Development passed on all three devices without changing the initial learning/matching settings. No claim is made that every scalar is blind: raw charge and the conditioned peak already discriminate histories. Offline whole-window conditioning, designed history locations, supervised calibration, and passive linear receiving dynamics are explicit limitations.

Fresh locked evaluation passed: 1,279/1,280 primary decisions correct, fixed/erased 50%, shuffled 50.7031%. A separate receipt audit exactly recomputed all primary voltages/decisions, verified training calibration/control decisions/shuffle permutations, regenerated representative histories/probe traces, and matched source/protocol hashes to the lock commit. All 49 tests pass. The figures were visually inspected. The offline conditioner and scalar diagnostics remain explicit limitations.
