# Reward experiment review

Independent read-only reviewer checked the new writer, gate, controls, costs, protocol boundary, tests, and replay before locking. All 40 tests passed independently. No critical or important implementation findings remained. No fresh seeds were run during review.

Resolved: initial documentation incorrectly claimed an all-seed replay selector. The existing renderer illustrates the first seed only. Documentation and reward replay now say so; the measurement table still aggregates all seeds.

Minor provenance finding: development ran from an uncommitted working tree, so its recorded base checkout commit does not reproduce the new source. This is explicitly documented. The full evaluation requires the exact clean committed lock and records its protocol hash and code commit. This review is implementation verification, not an independent replication or an endorsement of biological claims.

## Fresh result verification

The locked run completed all ten fresh seeds and failed only the accuracy threshold: reward 75%, shuffled reward 48.9%, random/frozen/erased 50%, teacher 96.25%. Exact recomputation matched 24,000 stored decisions and all motor energies (maximum difference 0), validated material constraints, matched the protocol hash to the lock commit, checked identical input digests across feedback arms, and confirmed solve counts. The result was not retuned.

All 40 tests passed again after the run. The static result chart was visually inspected; replay metadata, placeholders, reward wording, and JavaScript syntax were checked. Browser playback was not visually verified. The replay illustrates the first seed only.
