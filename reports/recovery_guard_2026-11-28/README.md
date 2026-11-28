# Recovery launcher guard diagnosis

The refusal was caused by the outer PowerShell launcher, not a live training
worker. `archive_stale_lock` checks every process for standalone `-m` followed
by `src.experiments.run` or `src.experiments.worker`. With `powershell -File
scripts/research_cpu.ps1 -m ...`, the outer PowerShell process contains those
same arguments. It is not one of the identical-command Python parents excluded
by the current guard, so the guard mistakes it for another experiment process.

The supplied restart command exposed this false positive. Frozen source and
the CPU launcher script have not been changed to repair it. Invoke the existing
CPU launcher through PowerShell's `-Command` form: its body is one command-line
argument, while the actual Python child retains its normal module arguments.
Real worker/scheduler processes remain subject to the original guard checks.

`probe_guard.py` reproduced the original refusal and identified `powershell.exe`
as the blocking ancestor. A second probe with the command wrapper passed all
production guard checks and reached the archive boundary. Both probes mocked
`Path.rename` to raise before any actual archival, so the lock remained unchanged
and no training was launched. Trailing `-m` arguments in the probe are inspection
data; the probe never dispatches the training module.

`file_wrapper.json` and `command_wrapper.json` retain the process/guard evidence.
`verify_resolution.py` reuses the earlier read-only outage inspection with a new
report destination, verifying all twenty-one completed-job seals, M1's sealed
epoch six and 113,532 optimizer steps, the dead lock, unchanged source/packages,
data and runtime contracts. The previous outage report is preserved.

## Corrected user command

Run this once from the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& './scripts/research_cpu.ps1' -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 6 --retry-failed --resume-supervised --recover-stale-lock"
```

This keeps the same D: CPU environment, logging and original guard, archives
the dead lock after its safety checks, restores M1 after epoch six and replays
epoch seven. It then completes M2–M6 at budget 18/seed 123, stopping at 27/48
completed jobs. Expect six jobs executed in this call and `limited_run: false`.
Send the summary before advancing. If the guard reports another process, report
the error rather than killing it or deleting the lock. Keep source/packages fixed.
