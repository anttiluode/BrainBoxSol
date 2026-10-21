# Independent local-slowness review

Base: main `6bce94ed7c9ab343a93747f966c71644ec8e1d6b`. The reviewer worked read-only and did not read saved development outcomes or run full seeds. Initial scientific source/development: `436feaec27ae13e3b272e70dcaf3fdd0febf99ef`. Refined development: `4df4e647a613d86967a283154b40524201a2b05f`.

Five Important findings were corrected before the source/protocol lock:

1. Delayed accuracy alone confounded initial writing and retention. The gain is now difference-in-differences, and both initial accuracies must be >=0.8.
2. Equal but poor replacement performance could pass a relative cost condition. Local replacement must now also be >=0.8.
3. Numerical refinement omitted trace, waiting and rewriting. It now checks both primary writers on actual training/test triplets through writing, primary waiting, replacement and subsequent waiting.
4. Source hashes were collected after execution. They are now snapshotted first, checked afterward, and raw results are saved before a separate replay audit.
5. Replay did not reject an unlocked protocol before considering full seeds. It now does so immediately.

Regression tests for these cases first failed on the archived implementation, then passed after fixes. The actual-training-triplet mismatch was also corrected. Minor issues addressed: passive RC operator validation, an overstated observation-hash promise, exact floating-byte hashing incompatible with a tolerant replay, and ambiguous refinement sampling counts.

The final re-review found no remaining Important issue. All 21 new tests passed, and the whole repository suite has 95 tests. No calibration leakage was found. The reviewer confirmed the paired nuisances, frozen train-only decoder, fast resets, equal material-state count, explicit resource controller and numerical sampling scope. Full-seed outcomes must follow this review and source lock; negative outcomes remain results.

The resulting claim is deliberately narrow: an engineered spatial trace with feedback may trade retention for rewriting in a supervised readout task. There is no claim about physical mass, quantum effects, emergent intelligence, digital efficiency, waveform necessity or repair of the earlier DC bridge failure.
