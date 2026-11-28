# CPU replication terminal handoff

The user selected the current native Windows CPU route on 20 November 2026.
The study is frozen at `runs/bilstm_replication_cpu_2026-11-20` with 48 jobs.
All six preparation/SSL jobs have completed and verified, including ten SSL
epochs for each of seeds 17/42/123. The first seven supervised models at budget
18/seed 17 also completed and verified. The seed-42 block now verifies as well:
twenty-one of forty-eight jobs are complete after M0/seed 123 finished. The
28 November outage interrupted M1/seed 123 with epoch six safely sealed.
Its [source protocol](../REPLICATION_PROTOCOL_CPU_2026-11-20.md)
fixes all seven models, both budgets and seeds 17, 42 and 123. Source/package,
participant-budget, protocol and recipe checks passed. Core source and the
environment remain unchanged from the accepted CPU engineering checks.

## Current outage restart

The original `powershell -File ... -m ...` restart triggered a guard false
positive on its own PowerShell parent. The
[read-only launcher diagnosis](../../reports/recovery_guard_2026-11-28/README.md)
confirmed the cause and tested the corrected `-Command` wrapper without training
or lock mutation. Frozen source and the CPU launcher remain unchanged.

The [28 November recovery inspection](../../reports/replication_power_loss_2026-11-28/README.md)
verified twenty-one completed-job seals, M1's six saved epochs and 113,532 saved
optimizer updates. The dead scheduler lock is preserved for automatic archival.
Run this once from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& './scripts/research_cpu.ps1' -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 6 --retry-failed --resume-supervised --recover-stale-lock"
```

This restores M1 after epoch six, replays epoch seven and continues through M6
at budget 18/seed 123. It stops before budget 70. Expect forty-eight planned,
twenty-seven completed, six executed and `limited_run: false`. The old summary
still says twenty because the outage prevented a new invocation summary. Do not
delete the lock or edit ledger/status files. Send the final summary before
advancing. Keep source/packages unchanged.

## Earlier seed 123 restart superseded by the outage command

The [seed-42 completion and interruption review](../../reports/replication_cpu_stage4_2026-11-27/README.md)
verifies twenty completed jobs, the new block's 52 epochs/59 recovery generations
and the interrupted M0/seed-123 epoch-zero checkpoint. Run this once:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 7 --retry-failed --resume-supervised
```

This preserves completed jobs, retries M0 in a fresh attempt from epoch zero,
then trains M1–M6 at budget 18/seed 123. It stops before budget-70 fits. Expect
forty-eight planned, twenty-seven completed, seven executed and `limited_run: false`.
The first epoch must be replayed because validation/checkpoint publication did
not finish. The stale `running` artifact status does not override the failed
ledger entry; preserve it for provenance. There is no scheduler lock to remove.
Send the summary before repeating or advancing. Keep source/packages unchanged.

## Completed seed 42 block

The [first supervised block review](../../reports/replication_cpu_stage3_2026-11-24/README.md)
verified seven models, 69 completed epochs, 76 recovery generations and shared
65,133-target validation exports. The user completed the following seed-42
command, retained for provenance:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 7
```

This trains M0 through M6 at label budget 18 and optimization seed 42, then stops
before seed-123 supervised fits. It uses the current study's seed-42 SSL encoder
for M4/M5/M6, not historical supervised checkpoints. Expect forty-eight planned,
twenty completed, seven executed this call and `limited_run: false`. That
summary has passed inspection. The preceding seed-17
block recorded 42.48 hours; the new seed's stopping times may differ.

## Completed first supervised block

The [SSL completion review](../../reports/replication_cpu_stage2_2026-11-22/README.md)
verifies all three encoders and the unchanged freeze. The user completed this
first supervised command; it is retained for provenance:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 7
```

This trains M0 through M6 at the 18-participant label budget and optimization
seed 17, then stops before seed-42 supervised fits. M4/M5/M6 use the completed
seed-17 SSL encoder. All fits retain the frozen schedule, epoch recovery and
eager batch-four validation. Expect forty-eight planned, thirteen completed,
seven executed this call and `limited_run: false`. That summary has now passed
inspection. The corresponding historical block recorded 33.39 hours
under the older procedure; that is context, not a new-run duration guarantee.

## Completed second stage

The [first-stage review](../../reports/replication_cpu_stage1_2026-11-20/README.md)
verifies all four completed jobs, both normalizers and the seed-17 SSL bundle
and final checkpoint hashes. The user completed this command; it is retained
for provenance:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 2
```

This verifies/skips the completed jobs, executes fresh ten-epoch SSL fits for
seeds 42 and 123, then stops before supervised training. Expect forty-eight
planned, six completed and two executed this call, with `limited_run: false`.
That 6/48 summary has been inspected successfully.

## Completed first stage

The user completed this first-stage command; it is retained for provenance:

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
