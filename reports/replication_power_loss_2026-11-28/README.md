# Replication power loss recovery inspection

Recovery is ready. The 28 November inspection verified all twenty-one completed
job seals and the interrupted model's seven sealed recovery generations without
TensorFlow imports, model execution or optimizer updates. Source, full installed
packages, data scope, protocol, launcher and recorded CPU execution environment
match the freeze. No files inside the frozen study were changed.

| Work | Preserved state |
|---|---|
| First twenty completed jobs | All seals verify |
| M0 at budget 18, seed 123 | Completed attempt 0003; eleven epochs; best epoch six |
| M1 at budget 18, seed 123 | Attempt 0001 interrupted; six complete epochs sealed |
| M1 optimizer | 113,532 updates saved |
| M1 best model | Epoch six, matching the latest generation's best-model copy |
| M1 unfinished work | Epoch-seven training finished; validation/publication did not finish |
| Scheduler lock | PID 94504 is dead; no visible experiment process found |

The earlier execution summary still reports twenty completed jobs because the
outage prevented a new invocation summary. The ledger and M0's completed job
seal establish that twenty-one jobs are now complete. M1's running ledger status
is stale after the outage. Do not edit statuses, remove the lock or alter
checkpoints manually.

M1 can restore the sealed end of epoch six and replay epoch seven, including
model and optimizer arrays, RNG/callback state, history and best-model selection.
Work after that sealed boundary is not saved. `inspect_recovery.py` verifies the
checkpoint's contract and counters and confirms the scheduler will select it.
Integrity/compatibility inspection is not an executed resume test; the earlier
exact real-data replay result applies to its tested capped M6 case.

## User restart command

Run this once from the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 6 --retry-failed --resume-supervised --recover-stale-lock
```

The command rechecks compatibility, archives the dead scheduler lock, skips
completed jobs and starts a fresh M1 attempt from epoch six. It then runs M2
through M6 at budget 18/seed 123 and stops before budget 70. Six additional jobs,
rather than seven, keep the original stage boundary. Prior attempts, checkpoint
generations and dead lock contents are preserved. If the PID is alive or another
experiment process appears, the lock guard refuses recovery; report the error.

Expect forty-eight planned, twenty-seven completed, six executed this call and
`limited_run: false`. Send the summary before advancing. Keep source, environment,
data and frozen manifests unchanged. The assistant neither archived the lock
nor launched training. No source-test or ISRUC predictive scoring was opened.
