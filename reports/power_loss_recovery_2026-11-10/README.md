# Power-loss inspection and recovery handoff — 10 November 2026

The user reported a second power interruption during the BiLSTM terminal study
`runs/bilstm_terminal_2026-11-02`. The assistant inspected artifacts and prepared
a restart handoff without launching training, loading TensorFlow or performing
optimizer updates.

## Verified progress

Fifteen of eighteen jobs are complete: three preparation jobs, ten-epoch SSL,
and eleven of fourteen supervised experiments. All seven conditions at the
18-participant budget and M0–M3 at the 70-participant budget are complete.
The source waveform fit and both budget-specific normalizers remain preserved.
SSL completed 101,060 optimizer iterations and does not need to be repeated.

| Labeled participants | Condition | Model / initialization | Epochs completed | Best epoch | Validation macro-F1 |
|---:|---|---|---:|---:|---:|
| 18 | M0 | Spectral BiLSTM | 7 | 2 | 0.655117 |
| 18 | M1 | Raw EEG / scratch | 10 | 5 | 0.726231 |
| 18 | M2 | Concatenation / scratch | 6 | 1 | 0.721672 |
| 18 | M3 | Gated cross-attention / scratch | 8 | 3 | 0.706983 |
| 18 | M4 | Raw EEG / SSL | 6 | 1 | 0.719047 |
| 18 | M5 | Concatenation / SSL | 7 | 2 | 0.731709 |
| 18 | M6 | Gated cross-attention / SSL | 8 | 3 | 0.733935 |
| 70 | M0 | Spectral BiLSTM | 12 | 7 | 0.686254 |
| 70 | M1 | Raw EEG / scratch | 16 | 11 | 0.773059 |
| 70 | M2 | Concatenation / scratch | 10 | 5 | 0.773037 |
| 70 | M3 | Gated cross-attention / scratch | 15 | 10 | 0.756461 |

All classifiers use BiLSTM temporal heads. Completed classifier histories cover
105 epochs and about 85.1 recorded elapsed hours; completed SSL adds about
1.5 hours, for about 86.6 hours of completed recorded work. This excludes
preparation and interrupted attempts and does not predict remaining runtime.
The counts are jobs/epochs, not percentages of total compute.

Compared with the 8 November inspection, M1, M2 and M3 at budget 70 have now
finished, preserving an additional 41 epochs. The old independent M1 snapshot
still matches its recorded hashes, but it is an intermediate epoch-six snapshot;
the completed M1 model's selected checkpoint is epoch eleven. The editor's open
snapshot normalizer file is not the latest model or a resume checkpoint.

All fifteen completed job inventories verify. For each completed classifier,
the review also checked bundle hashes, finite history values,
agreement of best epoch with validation macro-F1, Keras ZIP CRC integrity and
finiteness of stored model/optimizer tensors. Their model checkpoints,
normalizers, configurations, histories, validation metrics, predictions and
completion seals remain available in the original study attempt directories.

These are existing source-validation development results from one seed. They
are not new source-test or ISRUC results and do not establish statistical
significance or confirmatory transfer gains. No predictive evaluation ran in
this inspection.

## Interrupted experiment

The abandoned job is `M4_draw42_n70_seed42`, raw EEG with SSL initialization.
Its first training pass reached batch 79,807/79,807 in about 37 minutes
20 seconds. The log then entered source-validation computation and ended.
No validated epoch was published: history is zero bytes, and no `model.keras`
or completed-job seal exists. Thus no trained M4 weights or optimizer state
from this attempt can be restored after the outage. The log establishes where
the published progress stopped; it cannot establish exactly how far validation
advanced before power loss.

Only M4 needs to restart. Its next attempt will initialize from the already
completed, verified SSL encoder and use the existing normalizer. Its interrupted
attempt and logs remain preserved. M5 and M6 at budget 70 have not started.
Three supervised jobs remain to complete the matrix; their duration is not
known in advance because of early stopping.

## Safe restart preparation

`status_before.json` records the initial state: fifteen completed jobs, an
abandoned running ledger entry, two pending jobs and a dead scheduler PID.
`inspection.json` records the full checks, artifact locations and interrupted
file hashes. Source code, package versions and source-scope manifests match the
freeze; all ten historical baseline hashes and the November 8 snapshot hashes
also match. No experiment scheduler or worker was found.

After checking process absence, confirming the lock PID was dead and rechecking
the lock bytes, the assistant renamed `.runner.lock` to
`.runner.lock.power_loss_2026-11-10`. The old lock bytes, earlier archived lock,
ledger, source, packages, models and attempts were preserved. This removes the
abandoned lock without altering completed experiments or declaring M4 complete.

When ready, the user can run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_terminal_2026-11-02 --live --retry-failed
```

It verifies/skips fifteen completed jobs, retains M4 attempt 0001, starts M4
attempt 0002 from its declared SSL initialization, then runs M5 and M6. This is
continuation at the experiment level, not exact within-epoch resume. No new study
is required while implementation/packages remain unchanged. The assistant did
not execute this training command. Previously reported thermal throttling was
not remeasured by this artifact inspection.

The study-level summary and CSV tables still describe the older first-stage
execution because the interrupted scheduler never reached final table export.
Use verified per-job seals, the ledger and this inspection for current status.
