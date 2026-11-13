# Recovery/validation implementation handoff — 13 November 2026

The completed 18-job study was verified idle before source edits. New opt-in
supervised epoch recovery, compiled validation and a separate validation batch
size have been implemented. No research training, pretraining, fine-tuning,
optimizer profile or optimizer-based regression was run by the assistant.

`checks.txt` records 26 passing non-optimizer tests. These cover sealed numeric
state, corrupt files and changed contracts, failed pointer publication, preserved
prior generations, mock callback/variable/RNG restoration, latest-epoch selection,
safe stale-lock handling and CLI/worker ownership. Real model construction and
inference-only checks cover raw, concat, gated and spectral BiLSTMs, tail batches,
compiled/eager probability parity, and target/metric parity across inference
batch sizes. No test called a real model's fit or an optimizer update.

An initial added inference-batch fixture lacked the normalizer compatibility
method; that fixture was corrected and the full explicit non-optimizer group
then passed. This was a fixture error, not a failed optimizer experiment.
TensorFlow's cross-function tracing warning arises while testing multiple fresh
models; the per-model compiled predictor is checked to trace once across batch
and tail shapes.

`evidence.json` records current source hashes and preservation checks: all 18
historical job inventories and all ten baseline hashes still verify. The old
study's source hashes intentionally differ now; it remains completed historical
evidence and is not a resume target for this implementation.

The synthetic fresh-process optimizer gate and full regression suite are
prepared but not executed. Recovery equivalence and end-to-end speedup remain
unproven until the user-run gates/benchmarks pass. See
[the handoff guide](../../docs/operations/SUPERVISED_RECOVERY.md) for the first
command, configuration, preserved attempt semantics and strict resume limits.

No new source-test/ISRUC predictive scoring, package installation or cloud
upload occurred. Caches and test fixtures use the existing D: CPU launcher.
