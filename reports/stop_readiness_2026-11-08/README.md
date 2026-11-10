# Inspection before a possible training stop — 8 November 2026

The user asked what would survive stopping the current CPU campaign. Training
was left running. Only read-only inspection and new snapshots/reports outside
the active study were performed; no source, package, ledger, lock, checkpoint or
training setting was modified.

At the observation, 12 of 18 jobs were complete and verified, one was running,
and five had not started. Counts of jobs do not estimate remaining compute time.
`status.json` records the live scheduler and initial observation;
`inspection.json` records the subsequent artifact checks and active snapshot.

## Complete and preserved

The three preparation jobs are complete: quality-only waveform normalization
over 323,423 training epochs and spectral normalization for each labeled budget.
SSL completed ten epochs and 101,060 optimizer iterations; its encoder,
normalizer, diagnostics, history and checkpoint files remain sealed.

Every file in all twelve completed job inventories passes SHA-256 verification.
The eight completed classifiers are:

| Labeled participants | Condition | Model inputs / initialization | Epochs run | Best epoch | Validation macro-F1 |
|---:|---|---|---:|---:|---:|
| 18 | M0 | Spectral BiLSTM | 7 | 2 | 0.655117 |
| 18 | M1 | Raw EEG / scratch | 10 | 5 | 0.726231 |
| 18 | M2 | Concatenation / scratch | 6 | 1 | 0.721672 |
| 18 | M3 | Gated cross-attention / scratch | 8 | 3 | 0.706983 |
| 18 | M4 | Raw EEG / SSL | 6 | 1 | 0.719047 |
| 18 | M5 | Concatenation / SSL | 7 | 2 | 0.731709 |
| 18 | M6 | Gated cross-attention / SSL | 8 | 3 | 0.733935 |
| 70 | M0 | Spectral BiLSTM | 12 | 7 | 0.686254 |

All classifiers use the BiLSTM temporal head. These are source-validation
development scores from one seed, not test or external evaluation, confidence
intervals or confirmatory transfer claims. Epoch counts below thirty reflect
the frozen early-stopping policy. M5 at budget 18 is the successfully completed
second attempt; its first power-interrupted attempt remains preserved too.

Each completed classifier retains its best model, normalization/configuration,
data provenance, complete history, exported validation probabilities and epoch
identities, validation metrics/confusion matrix, summary and completion seal.
These jobs do not need to be rerun when using the same unchanged study.

The eight completed classifiers account for about 37.8 recorded elapsed hours;
completed SSL adds about 1.5 hours. These totals exclude the active classifier,
preparation, and interrupted attempts and are not an estimate of remaining time.

## Active model: retained checkpoint versus exact resume

`M1_draw42_n70_seed42` is the raw-EEG model trained from scratch at budget 70.
At the inspection it had seven fully validated/logged epochs and was training
epoch eight. The latest sampled batch was 3,800 of 79,807 for that epoch. Live
progress naturally advances after this observation.

Its best saved checkpoint corresponds to epoch six (source-validation macro-F1
0.770410). The Keras archive passes ZIP CRC checks; all 74 stored tensors are
finite. It contains optimizer state with recorded iteration 478,842 and learning
rate approximately 0.0005. Iterations match six passes of 79,807 batches.
This confirms stored state, not that it is a complete exact-resume checkpoint.

A stable independent snapshot was copied to:

`runs/stop_readiness_2026-11-08/active_model_snapshot/`

It contains `model.keras`, `model_config.json`, `normalizer.json`,
`data_manifest.json`, `resolved_config.json`, `training_history.csv`, and
`status.json`. Source sizes/write timestamps were unchanged throughout the
read, and copied bytes match recorded SHA-256 hashes. The snapshot's status
still says running because it is an observation, not a completed model bundle.
It protects this observed best checkpoint even if the live checkpoint is later
replaced or interrupted while writing. Original attempt logs remain in place.

The active model has no completed-job seal, final inference bundle or final
exported validation predictions yet. It must not be counted as a completed
experiment. Parameters from epoch seven or the in-progress epoch eight are not
separately saved: best-only checkpointing retains epoch six at this observation.

The existing supervised launcher does not restore the epoch/sampler sequence,
early-stopping state or learning-rate callback state. `execute --retry-failed`
would preserve the interrupted attempt and start a new M1 attempt from scratch,
skipping the twelve completed jobs. Loading the saved weights alone, even with
stored optimizer variables, is not equivalent to that missing exact-resume
procedure. A future checked resume implementation would require separate
development and provenance handling; none was applied to this frozen campaign.

## Not started and historical preservation

M2, M3, M4, M5 and M6 at budget 70 have not started. They will not be lost if
training stops; they simply remain pending. Completing the full budget comparison
still requires the current M1 job and these five jobs.

All ten historical baseline files listed in
`reports/implementation_2026-10-27/evidence.json` still match their recorded
hashes. Source implementation, package versions and the frozen participant/data
scope also match. No historical run was restarted and no source-test/ISRUC
predictive evaluation was performed. Reading existing validation summaries and
checkpoint tensors required no TensorFlow import or optimizer operation.

## If the user chooses to stop

Use Ctrl+C in the training terminal and wait for the scheduler to exit. The live
runner is designed to terminate its owned worker tree, retain attempts, record
the interruption and release its lock. This inspection did not issue Ctrl+C or
stop any process. A completed job is preserved independently of the current
worker's state. Inspect status after stopping before any later restart.
