# First supervised replication block review

The user completed M0 through M6 at label budget 18 and optimization seed 17.
The 24 November review verified all thirteen completed-job seals and the
unchanged source, installed packages, data scope, recipe and archived protocol.
The seven supervised models completed 69 epochs in 42.48 recorded hours.
All 76 recovery generations load with valid seals, finite state arrays,
consistent history and optimizer counters. Selected best-model copies match
the completed model files.

All seven prediction exports contain the same 65,133 source-validation target
IDs, labels and order. Recomputed metrics match the saved summaries and selected
history rows. Normalization is identical within the block; M4/M5/M6 use this
study's seed-17 SSL encoder. Each model retained the frozen uncapped schedule,
epoch recovery and eager batch-four validation. No model was run by the reviewer.

| Model | Completed epochs | Selected epoch | Pooled validation macro-F1 |
|---|---:|---:|---:|
| M0 | 9 | 4 | 0.650148 |
| M1 | 9 | 4 | 0.699695 |
| M2 | 7 | 2 | 0.718853 |
| M3 | 10 | 5 | 0.721021 |
| M4 | 10 | 5 | 0.724610 |
| M5 | 10 | 5 | 0.717318 |
| M6 | 14 | 9 | 0.719440 |

These are checkpoint-selection metrics for one optimization seed and one label
budget, not independent held-out evidence or the protocol's final participant-
averaged multi-seed contrasts. Keep the frozen family and analysis unchanged;
do not select conditions or revise the recipe from this block's rankings.
Recovery file integrity does not itself prove resumed full-schedule equivalence.
The 42.48 hours include the fits' recorded loading, validation and final-export
work; this is not an isolated optimizer throughput or speedup measurement.

`inspect_stage.py` reproduces the checks without TensorFlow imports, model
execution or optimizer updates. `verification.json` records job seals, recovery
counts and the next stage; `source_validation_summary.csv` retains precise
per-model values. Historical runs and this completed block remain preserved.
No source-test or ISRUC predictive scores were opened.

## Next user run

Run this once from the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 7
```

The runner skips the thirteen completed jobs and trains M0 through M6 for
`draw42_n18`, optimization seed 42. M4/M5/M6 use the new study's completed SSL
seed 42. This is not a reuse of the historical seed-42 supervised models. The
command stops before seed-123 supervised fits. Expect forty-eight planned,
twenty completed and seven executed in this call, with `limited_run: false`.

Send that summary before repeating or advancing; the job limit counts additional
jobs on each invocation. Keep source, packages, data and frozen manifests
unchanged. On an interruption, preserve attempts, checkpoints, logs and lock
files and request inspection before the explicit supervised retry/resume command.
The assistant has not launched the next block.
