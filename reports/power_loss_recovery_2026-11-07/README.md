# Power-loss inspection and restart handoff — 7 November 2026

The user reported an abrupt laptop shutdown during native CPU execution of
`runs/bilstm_terminal_2026-11-02`. No training was launched during this review.

Nine of eighteen planned jobs are complete and their sealed inventories verify:
three preparation jobs, ten-epoch source-only SSL, and five supervised models at
the 18-participant label budget. These are counts of jobs, not percentages of
total compute or expected remaining duration.

| Completed classifier | Representation / initialization | Completed epochs | Best epoch |
|---|---|---:|---:|
| M0 | Spectral BiLSTM | 7 | 2 |
| M1 | Raw EEG, supervised from scratch | 10 | 5 |
| M2 | Concatenation, supervised from scratch | 6 | 1 |
| M3 | Gated cross-attention, supervised from scratch | 8 | 3 |
| M4 | Raw EEG, SSL initialization | 6 | 1 |

The interrupted job is `M5_draw42_n18_seed42`: concatenation with SSL
initialization. Its log ends after batch 18,922/18,922 of epoch one, followed
by the start of source-validation metrics. No validated epoch was published:
`training_history.csv` is empty and no `model.keras` exists. The current trainer
saves its best checkpoint after validation, so this first training pass has no
recoverable model/optimizer state. Restart M5 from the verified completed SSL
encoder, retaining its old attempt. Eight jobs have not started: M6 at budget
18 and all seven conditions at budget 70.

`status.json` preserves the initial observation: the ledger still recorded M5
as running, but the scheduler lock PID was dead. `inspection.json` records
completed artifact checks, epoch counts and recovery details. All source hashes,
package versions and source-scope checks pass. No experiment scheduler or worker
was found. After rechecking process absence and lock bytes, the abandoned lock
was renamed to `.runner.lock.power_loss_2026-11-07` in the study directory.
Its bytes, the ledger, models, old attempts, source and packages were preserved.

The user can restart the remaining work in their terminal:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_terminal_2026-11-02 --live --retry-failed
```

The scheduler verifies/skips nine completed jobs, preserves M5 attempt 0001,
starts M5 attempt 0002 from its declared SSL initialization, then advances to
the eight pending jobs. `--retry-failed` also handles abandoned ledger entries
marked `running`; it does not enable within-job checkpoint resume. Older study
summary/CSV files may still describe the first four jobs because the interrupted
execution did not reach the final table-writing stage. Use the verified ledger
and this inspection for current progress.

This inspection imported no TensorFlow and performed no optimizer updates,
model fitting, ISRUC scoring or new source-test scoring.
