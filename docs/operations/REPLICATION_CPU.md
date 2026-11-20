# CPU replication terminal handoff

The user selected the current native Windows CPU route on 20 November 2026.
The new study is frozen at `runs/bilstm_replication_cpu_2026-11-20` with 48 jobs
and no jobs started. Its [source protocol](../REPLICATION_PROTOCOL_CPU_2026-11-20.md)
fixes all seven models, both budgets and seeds 17, 42 and 123. Source/package,
participant-budget, protocol and recipe checks passed. Core source and the
environment remain unchanged from the accepted CPU engineering checks.

## First stage

Run this once from the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 4
```

This executes waveform statistics, the 18-participant normalizer, the
70-participant normalizer and `ssl_seed17` for its full ten epochs. It stops
before SSL seeds 42/123 or any supervised model. The expected summary is:

```json
{
  "jobs_planned": 48,
  "jobs_completed": 4,
  "jobs_executed_this_call": 4,
  "limited_run": false
}
```

The 4/48 summary is a successful first stage, not completion of the whole study.
`limited_run: false` describes uncapped job schedules; `--max-jobs 4` limits the
number of additional jobs in this invocation. Repeating it advances another
four jobs. Send the first-stage summary for inspection before repeating or
continuing. There is no background controller or automatic next-stage launch.

## Monitoring and preservation

Live output shows progress. Worker stdout/stderr is also captured directly in
`runs/bilstm_replication_cpu_2026-11-20/jobs/<job>/attempt_0001.log`, with
artifacts/checkpoints in the sibling attempt directory. This avoids reliance
on the earlier incomplete PowerShell transcript. Project-managed caches and
temporary files continue to use `runs/cpu_runtime` on D:.

A read-only status command can run from a second terminal:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 scripts/training_status.py --study-dir runs/bilstm_replication_cpu_2026-11-20 --verify-artifacts --verify-data-scope
```

For this new study, require source/package and data-scope checks to match.
Preserve `src/`, the environment, data, frozen manifests and the archived
`protocol.md`/`protocol_freeze.json`. Do not run the older completed study or
reuse historical seed-42 results as this study's third seed.

## Interruption handling

If power fails or a job errors, preserve its attempt and logs and request an
inspection before retrying. The supervised epoch-resume interface is documented
in [SUPERVISED_RECOVERY.md](SUPERVISED_RECOVERY.md). Its flag does not add
scheduler-level SSL resume. An interrupted SSL job's sealed epochs should be
preserved for inspection rather than assuming `--resume-supervised` will recover
them. Never delete locks or bypass source/package/contract checks to continue.

The measured 59.47-minute benchmark capped training at 64 batches and included
two full validation passes; it is not the full campaign duration. The historical
single-seed matrix recorded about 118 hours, which is context rather than a
new-version forecast. No new speedup, thermal safety or full-schedule recovery
guarantee is implied by these measurements.
