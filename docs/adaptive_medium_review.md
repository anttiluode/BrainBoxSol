# Adaptive material memory: independent review record

Native implementation was followed by one fresh whole-branch review using
gpt-6-astra. The reviewer inspected the approved specification, implementation
plan, production code, tests, CI, report, measured receipt and replay.

The reviewer independently confirmed all 31 original tests, exact regeneration
of learned/frozen/traffic material responses across all ten held-out seeds,
the stored summary and protocol digest, and all replay episodes. There were
no Critical issues and no deferred Minor findings.

## Important findings and fixes

1. **Resume trusted saved seed records after checking only provenance.** A
   corrupted material vector could retain stale successful metrics. Resume
   now checks all material bounds/budgets and initial state, complete mappings,
   arms/counts, classifier parameters and numerical records, and regenerates
   all material and shuffle response evidence before reusing any seed. The
   complete CLI regression fails on the original lock code and passes after
   the fix. No extra held-out training was performed to make this regression.
2. **Replay input validation omitted fields used by the browser.** A missing
   energy array could crash playback; a missing selection could display the
   first episode instead. The renderer now validates the full episode fields
   and condition coverage, offers only recorded mappings, and never silently
   substitutes a different episode. Missing-energy and incomplete-condition
   regressions fail before the fix and pass afterwards.

All 33 tests pass after the single fix pass. The original measured full JSON
receipt is unchanged and retains producing commit
`032974c6269d54ebe24c1c3ec0e12ffa07028ea6`. Replay HTML was regenerated with
the safer wrapper. The fixes affect recovery and presentation validation,
not the fresh-run learning dynamics or the original scientific verdict.

## Decisions and review boundaries

- Shuffled-control receipts retain exact per-permutation metrics and energy
  summaries plus deterministic seeds, rather than every individual shuffle
  trace. A specific shuffled example requires deterministic regeneration.
- Interactive browser layout remains unverified because no browser executable
  was installed. Data round-trips, escaping, JavaScript syntax and a static
  figure of actual measurements were checked; browser UI defects may remain.
- Biological plausibility, open-ended learning and digital-AI superiority
  are untested broader claims. The accepted claim is supervised material
  memory on this fixed graph task.
- Committed provenance supports the recorded protocol and execution. Code
  inspection cannot establish the absence of unknown historical human tuning.

No independent re-review was commissioned: both identified issues received
regression tests and the complete suite was rerun. The PR remains open for
human review; no earlier PR or branch was merged by this work.
