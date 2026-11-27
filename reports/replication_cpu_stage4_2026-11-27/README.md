# Seed 42 supervised block and interruption review

The 27 November review verifies the seven completed models at label budget 18
and optimization seed 42. All twenty completed-job seals match the frozen study,
including the earlier preparation/SSL and seed-17 models. Source, installed
packages, data scope, recipe, launcher and archived protocol remain unchanged.

The new block completed 52 epochs and 59 sealed recovery generations in 31.62
recorded hours. Every generation loads with finite numeric arrays, consistent
epoch histories, optimizer counters and contracts. Selected best-model copies
match completed model files. The same normalizer is shared across the block,
and M4/M5/M6 initialize from this study's seed-42 SSL fit.

| Model | Completed epochs | Selected epoch | Pooled validation macro-F1 |
|---|---:|---:|---:|
| M0 | 7 | 2 | 0.655117 |
| M1 | 10 | 5 | 0.726231 |
| M2 | 6 | 1 | 0.721672 |
| M3 | 8 | 3 | 0.706983 |
| M4 | 6 | 1 | 0.719047 |
| M5 | 7 | 2 | 0.731709 |
| M6 | 8 | 3 | 0.733935 |

All seven exports share the 65,133 target IDs, labels and order checked against
the preceding seed-17 cohort. Recomputed metrics match summaries and selected
history rows. These pooled source-validation scores selected checkpoints; they
are not held-out confirmation or the final participant-averaged three-seed
contrasts. Keep the recipe and family unchanged. Historical seed-42 results
remain separate evidence; no broad procedure-equivalence claim is made here.

`inspect_stage.py` records these checks in `verification.json` and precise
per-model values in `source_validation_summary.csv`. The review imports no
TensorFlow and performs no model execution or optimizer updates. Recovery
file integrity does not itself establish resumed full-schedule equivalence.
No source-test or ISRUC predictive scoring was opened.

## Additional interrupted job

The execution summary correctly reports twenty completed jobs. The current
ledger additionally records `M0_draw42_n18_seed123` attempt one as failed with
`KeyboardInterrupt`. Its log shows the first training pass finished and source
validation began, but no epoch-one checkpoint was published. Its artifact
`status.json` still says `running`; that is stale child status, while the
scheduler ledger records interruption. No scheduler lock or visible experiment
process was found during inspection. The reviewer did not stop a process or
change either status file.

The sealed epoch-zero generation loads successfully with zero optimizer steps,
empty history and matching seed-123 training, normalization and source contracts.
There is no saved first-epoch progress to recover. The retry must replay epoch
one from the preserved initial state. All twenty completed jobs remain intact;
the old failed attempt, log and checkpoint must be retained.

## Next user run

Run this once from the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 7 --retry-failed --resume-supervised
```

This verifies/skips the twenty completed jobs, retries M0 at budget 18/seed 123
in a fresh attempt with the sealed epoch-zero source, then executes M1 through
M6 for that same budget and seed. It stops before the 70-participant budget.
There is no stale lock to recover, so no lock-removal action is needed. No
training was launched by the assistant.

On success expect forty-eight planned, twenty-seven completed, seven executed
in this call and `limited_run: false`. Send the summary before repeating or
advancing. Keep source, packages, data and frozen manifests unchanged. If another
interruption occurs, preserve all outputs and request inspection before retrying.
